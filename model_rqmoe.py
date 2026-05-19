import torch
from torch import nn
from utils import assign_to_codebook, assign_batch_multiple


class RQMoEStep(nn.Module):
    def __init__(self, d, K, N, L, H, dropout):
        super().__init__()
        self.d, self.K, self.N, self.L, self.H = d, K, N, L, H

        self.codebook = nn.Embedding(K, d)
        self.instruction = nn.Embedding(K, d)
        self.instruction.weight.data.zero_()

        self.MLPconcat = nn.Linear(d + d, d)
        self.gate = nn.Sequential(
            nn.Linear(d, H, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, N, bias=False),
            nn.Softmax(dim=-1),
        )

        self.experts = []
        for n in range(N):
            layers = []
            for l in range(L):
                layer = nn.Sequential(
                    nn.Linear(d, H, bias=False),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(H, d, bias=False),
                )
                layers.append(layer)
            expert = nn.ModuleList(layers)
            self.add_module(f"expert{n}", expert)
            self.experts.append(expert)

    def encode(self, residual, instruct):
        zqs = self.codebook.weight
        K, d = zqs.shape
        bs, _ = instruct.shape

        zqs_r = zqs.unsqueeze(0).expand(bs, K, d).reshape(bs * K, d)
        ins_r = instruct.unsqueeze(1).expand(bs, K, d).reshape(bs * K, d)
        cc = self.MLPconcat(torch.cat((zqs_r, ins_r), dim=1))

        expert_weights = self.gate(cc)
        expert_outputs = []
        for expert in self.experts:
            x = cc
            for layer in expert:
                x = x + layer(x)
            expert_outputs.append(x)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        weighted_expert_output = (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=1)
        if self.training:
            self.auxiliary_loss = (expert_weights.mean(dim=0) ** 2).sum() * 0.01

        zqs_r = weighted_expert_output.reshape(bs, K, d)
        codes, quantized = assign_batch_multiple(residual, zqs_r)

        ins = self.instruction.weight[codes]
        return codes, quantized, ins

    def decode(self, codes, instruct):
        zqs = self.codebook(codes)
        cc = self.MLPconcat(torch.cat((zqs, instruct), 1))
        expert_weights = self.gate(cc)
        expert_outputs = []
        for expert in self.experts:
            x = cc
            for layer in expert:
                x = x + layer(x)
            expert_outputs.append(x)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        weighted_expert_output = (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=1)
        if self.training:
            self.auxiliary_loss = (expert_weights.mean(dim=0) ** 2).sum() * 0.01

        toadd = weighted_expert_output
        ins = self.instruction.weight[codes]
        return toadd, ins


class RQMoE(nn.Module):
    def __init__(self, d, K, M, N, L, H, dropout):
        super().__init__()
        self.d, self.K, self.M, self.N, self.L, self.H = d, K, M, N, L, H

        self.codebook0 = nn.Embedding(K, d)
        self.instruction0 = nn.Embedding(K, d)
        self.instruction0.weight.data.zero_()

        self.steps = []
        for m in range(1, M):
            step = RQMoEStep(d, K, N, L, H, dropout=dropout)
            self.add_module(f"step{m}", step)
            self.steps.append(step)

    def encode(self, x):
        bs, _ = x.shape
        codes = torch.zeros(bs, self.M, dtype=torch.long, device=x.device)

        code0 = assign_to_codebook(x, self.codebook0.weight)
        codes[:, 0] = code0

        x_hat = self.codebook0.weight[code0]
        residual = x - x_hat
        instruct = self.instruction0.weight[code0]

        for i, step in enumerate(self.steps):
            codes[:, i + 1], toadd, ins = step.encode(residual, instruct)
            x_hat = x_hat + toadd
            instruct = instruct + ins
            residual = x - x_hat

        return codes, x_hat

    def decode(self, codes):
        x_hat = self.codebook0.weight[codes[:, 0]]
        instruct = self.instruction0.weight[codes[:, 0]]
        for i, step in enumerate(self.steps):
            toadd, ins = step.decode(codes[:, i + 1], instruct)
            x_hat = x_hat + toadd
            instruct = instruct + ins

        return x_hat

    def decode_parallel(self, codes):
        device = codes.device
        num_steps = len(self.steps)

        x_hat = self.codebook0.weight[codes[:, 0]]
        instruct_base = self.instruction0.weight[codes[:, 0]]

        if num_steps == 0:
            return x_hat

        step_codes_all = codes[:, 1:].T
        instruction_deltas = [
            step.instruction.weight[step_codes_all[i]]
            for i, step in enumerate(self.steps)
        ]
        instruction_deltas_tensor = torch.stack(instruction_deltas, dim=0)
        cumulative_deltas = torch.cumsum(instruction_deltas_tensor, dim=0)
        cumulative_instructions = instruct_base.unsqueeze(0) + cumulative_deltas
        all_instructions = torch.cat(
            [instruct_base.unsqueeze(0), cumulative_instructions], dim=0
        )

        step_codes_list = [step_codes_all[i] for i in range(num_steps)]
        step_instructions_list = [all_instructions[i] for i in range(num_steps)]
        if device.type == "cuda" and num_steps > 1:
            streams = [torch.cuda.Stream() for _ in range(num_steps)]
            step_outputs_tensors = [None] * num_steps

            for i, (step, step_codes, step_instruct) in enumerate(
                zip(self.steps, step_codes_list, step_instructions_list)
            ):
                with torch.cuda.stream(streams[i]):
                    toadd, _ = step.decode(step_codes, step_instruct)
                    step_outputs_tensors[i] = toadd

            for stream in streams:
                stream.synchronize()

            step_outputs_tensor = torch.stack(step_outputs_tensors, dim=0)
        else:
            step_outputs_list = []
            for step, step_codes, step_instruct in zip(
                self.steps, step_codes_list, step_instructions_list
            ):
                toadd, _ = step.decode(step_codes, step_instruct)
                step_outputs_list.append(toadd)
            step_outputs_tensor = torch.stack(step_outputs_list, dim=0)

        x_hat = x_hat + step_outputs_tensor.sum(dim=0)
        return x_hat

    def forward(self, x):
        with torch.no_grad():
            codes, _ = self.encode(x)

        losses = torch.zeros(self.M, device=x.device)
        x_hat = self.codebook0(codes[:, 0])
        instruct = self.instruction0(codes[:, 0])
        losses[0] = ((x_hat - x) ** 2).sum()

        for i, step in enumerate(self.steps):
            toadd, ins = step.decode(codes[:, i + 1], instruct)
            x_hat = x_hat + toadd
            instruct = instruct + ins
            losses[i + 1] = ((x_hat - x) ** 2).sum()

        return codes, x_hat, losses

    def freeze_codebooks(self):
        self.codebook0.weight.requires_grad = False
        for step in self.steps:
            step.codebook.weight.requires_grad = False
        print("All codebook weights frozen (requires_grad=False)")

    def unfreeze_codebooks(self):
        self.codebook0.weight.requires_grad = True
        for step in self.steps:
            step.codebook.weight.requires_grad = True
        print("All codebook weights unfrozen (requires_grad=True)")
