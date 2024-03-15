import numpy as np

def npy_to_txt(input_file, output_file):
    # Load the matrix from the .npy file
    matrix = np.load(input_file)
    
    # Save the matrix to a .txt file
    with open(output_file, 'w') as f:
        for row in matrix:
            # Convert each row to a string with each number formatted to integers
            row_str = ' '.join(f"{val}" for val in row)
            f.write(row_str + '\n')

if __name__ == "__main__":
    input_file = 'graph_permuted.npy'
    output_file = 'graph_permuted.txt'
    
    npy_to_txt(input_file, output_file)
    print(f"Matrix has been successfully written to {output_file}.")
