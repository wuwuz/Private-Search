import numpy as np
import random

def read_txt_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines

def write_txt_file(filename, lines):
    with open(filename, 'w') as file:
        file.writelines(lines)

def permute_txt_files(input_file_name, output_file_name, indices):
    lines = read_txt_file(input_file_name)
    permuted_lines = [lines[i] for i in indices]
    write_txt_file(output_file_name, permuted_lines)

def permute_npy_files(input_file_name, output_file_name, indices):
    matrix = np.load(input_file_name)
    permuted_matrix = matrix[indices]
    np.save(output_file_name, permuted_matrix)

def relable_txt_files(input_file_name, output_file_name, indices):
    # read the file where each row has multiple integers
    lines = read_txt_file(input_file_name)
    # make a list of lists of integers
    lines = [list(map(int, line.strip().split())) for line in lines]
    # relabel the integers in lines, where each integer i is replaced by indices[i]
    relabeled_lines = [[indices[i] for i in line] for line in lines]
    # convert the list of lists back to a list of strings
    relabeled_lines = [' '.join(map(str, line)) + '\n' for line in relabeled_lines]
    # write the relabeled lines to the output file
    write_txt_file(output_file_name, relabeled_lines)

def relabel_npy_files(input_file_name, output_file_name, indices):
    matrix = np.load(input_file_name)
    relabeled_matrix = np.array([indices[i] for i in matrix])
    np.save(output_file_name, relabeled_matrix)

def main():

    # Filenames
    docid = np.load("msmarco_docid.npy")
    n  = len(docid)
    indices = list(range(n))

    permute_npy_files("msmarco_docid.npy", "msmarco_docid_permuted.npy", indices)
    permute_npy_files("msmarco_embeddings_reduced.npy", "msmarco_embeddings_reduced_permuted.npy", indices)
    permute_txt_files("msmarco_embeddings_reduced.txt", "msmarco_embeddings_reduced_permuted.txt", indices)
    relabel_npy_files("msmarco-1280-cluster-reps.npy", "msmarco-1280-cluster-reps-relabeled.npy", indices)
    # save the indices
    np.save("permuted_indices.npy", indices)

if __name__ == "__main__":
    main()
