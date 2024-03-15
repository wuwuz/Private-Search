import requests

class VertexInfo:
    # a list containing the indices of the neighbors of the vertex
    def __init__(self, vector, neighbors):
        self.vector = vector
        self.neighbors = neighbors

def query_rows(row_indices):
    query_string = '&'.join([f'rowIndex={idx}' for idx in row_indices])
    response = requests.get(f'http://localhost:8080/query?{query_string}')

    VertexInfoList = []

    if response.status_code == 200:
        data = response.json()
        for item in data:
            print(f"Row: {item['matrixRow']}, Neighbors: {item['neighbors']}")
            VertexInfoList.append(VertexInfo(item['matrixRow'], item['neighbors']))
        return VertexInfoList
    else:
        print("Error:", response.status_code)
        return None

if __name__ == '__main__':
    list = query_rows([0, 1])  # Example query for rows 0 and 1
    print(list[0].vector)
    print(list[0].neighbors)
