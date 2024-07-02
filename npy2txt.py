import numpy as np

def npy_to_txt(input_file, output_file):
    # Load the matrix from the .npy file
    matrix = np.load(input_file)
    
    # Save the matrix to a .txt file
    with open(output_file, 'w') as f:
        for row in matrix:
            # Convert each row to a string with each number formatted to 3 decimal places
            row_str = ' '.join(f"{val:.5f}" for val in row)
            f.write(row_str + '\n')

if __name__ == "__main__":
    #input_file = 'input.npy'
    #output_file = 'output.txt'
    #input_file = 'msmarco_embeddings_reduced.npy'
    #output_file = 'msmarco_embeddings_reduced.txt'
    input_file = "msmarco-queries-1000-embeddings.npy"
    output_file = "msmarco-queries-1000-embeddings.txt"
    
    npy_to_txt(input_file, output_file)
    print(f"Matrix has been successfully written to {output_file} with numbers formatted to 5 decimal places.")
