#!/usr/bin/env python3
"""
Process .bvecs file to keep only the first N vectors.
Usage: python process_bvecs.py input.bvecs output.bvecs [num_vectors]
"""

import sys
import struct
import numpy as np

def read_bvecs_header(f):
    """Read dimension from first vector"""
    dim_bytes = f.read(4)
    if len(dim_bytes) != 4:
        return None
    dim = struct.unpack('<I', dim_bytes)[0]
    return dim

def process_bvecs(input_path, output_path, num_vectors=1000000):
    """
    Process .bvecs file to keep only first num_vectors.
    
    .bvecs format:
    - Each vector: 4 bytes (dim as uint32, little-endian) + dim bytes (uint8 values)
    """
    print(f"Processing {input_path} -> {output_path}")
    print(f"Keeping first {num_vectors:,} vectors")
    
    with open(input_path, 'rb') as fin:
        dim_bytes = fin.read(4)
        if len(dim_bytes) != 4:
            raise ValueError("Cannot read dimension from file")
        dim = struct.unpack('<I', dim_bytes)[0]
        print(f"Vector dimension: {dim}")
        
        # Calculate bytes per vector: 4 (dim) + dim (uint8 values)
        bytes_per_vector = 4 + dim
        
        # Seek back to start
        fin.seek(0)
        
        with open(output_path, 'wb') as fout:
            vectors_written = 0
            while vectors_written < num_vectors:
                vector_data = fin.read(bytes_per_vector)
                if len(vector_data) < bytes_per_vector:
                    print(f"Reached end of file at {vectors_written:,} vectors")
                    break
                
                fout.write(vector_data)
                vectors_written += 1
                
                if (vectors_written + 1) % 100000 == 0:
                    print(f"  Processed {vectors_written + 1:,} vectors...", end='\r')
            
            print(f"\nDone! Wrote {vectors_written:,} vectors to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_bvecs.py <input.bvecs> <output.bvecs> [num_vectors]")
        print("  num_vectors defaults to 1,000,000")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    num_vectors = int(sys.argv[3]) if len(sys.argv) > 3 else 1000000
    
    try:
        process_bvecs(input_path, output_path, num_vectors)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
