#include "main.h"
#include <chrono>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream input_file;
    std::string filename = argv[1];

    input_file.open(filename);
    if (!input_file.is_open())
    {
        std::cerr << "fileopen error " << filename << std::endl;
        return 1;
    }

    int N;
    input_file >> N;

    std::vector<int> input(N), output(N);
    for (int i = 0; i < N; ++i)
        input_file >> input[i];
    input_file.close();

    // auto start = std::chrono::high_resolution_clock::now();
    solve(input.data(), output.data(), N);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Kernel time: " << duration.count() << "us" << std::endl;
    solve(input.data(), output.data(), N);

    for (int i = 0; i < N; ++i)
        std::cout << output[i] << " ";
    // printf("%d ", output[i]);
    // std::cout << std::endl;
    std::cout << std::endl;
}
