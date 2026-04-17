#include "psrdada.h"

PSRDADA::PSRDADA(string filename)
{
    infileName = filename;
    infile.open(filename, ios::binary | ios::ate);
    if (!infile.is_open())
    {
        cout << "Cannot open file: " << filename << endl;
        exit(1);
    }
    streampos pos = infile.tellg();
    filesize = static_cast<unsigned long>(pos);
    infile.seekg(0, ios::beg);
    outfileName = filename.substr(0, filename.find_last_of('.')) + "_dedisp_folded.txt";

    // 目前只考虑只有一个4096字节的header的情况
    infile.read(headerBuffer, blockSize);
    string header(headerBuffer);
    header.erase(std::remove(header.begin(), header.end(), '\0'), header.end());

    string line;
    istringstream iss(header);
    while (getline(iss, line))
    {
        // cout << line << endl;
        auto parts = splitByMultipleSpaces(line);
        if (verbose)
            cout << parts.first << " " << parts.second << endl;
        if (parts.first == "BW")
        {
            bw = stof(parts.second);
        }
        else if (parts.first == "FREQ")
        {
            fc = stof(parts.second);
        }
        else if (parts.first == "NBIT")
        {
            nbit = stoi(parts.second);
        }
        else if (parts.first == "NCHAN")
        {
            nchan = stoi(parts.second);
        }
        else if (parts.first == "NDIM")
        {
            ndim = stoi(parts.second);
        }
        else if (parts.first == "NPOL")
        {
            npol = stoi(parts.second);
        }
    }
    cout << bw << " " << fc << " " << nbit << " " << nchan << " " << ndim << " " << npol << endl;
}

PSRDADA::~PSRDADA()
{
    infile.close();
}

void PSRDADA::initBuffer(size_t n)
{
    if (n % frameSamples != 0)
    {
        cout << "n must be a multiple of frameSamples (2048)" << endl;
        exit(1);
    }
    pol1_in = new uint16_pair[n];
    pol2_in = new uint16_pair[n];
    frameCount = n / frameSamples;
    if (verbose)
    {
        cout << "readSize: " << n << endl;
        cout << "frameCount: " << frameCount << endl;
        cout << "Filesize: " << filesize << " bytes" << endl;
    }
    readCount = static_cast<unsigned long>((filesize - blockSize) / (2 * n * sizeof(uint16_pair)));
    if (verbose)
    {
        cout << "readCount: " << readCount << endl;
        line();
    }
}

void PSRDADA::readSamples()
{
    for (int i = 0; i < frameCount; i++)
    {
        infile.read((char *)pol1_in + i * frameSamples * sizeof(uint16_pair), frameSamples * sizeof(uint16_pair));
        infile.read((char *)pol2_in + i * frameSamples * sizeof(uint16_pair), frameSamples * sizeof(uint16_pair));
    }
}

pair<string, string> PSRDADA::splitByMultipleSpaces(const string &input)
{
    size_t pos = input.find_first_of(" ");
    if (pos == string::npos)
    {
        return {input, ""};
    }

    size_t start = input.find_first_not_of(" ", pos);
    if (start == string::npos)
    {
        return {input.substr(0, pos), ""};
    }

    string part1 = input.substr(0, pos);
    string part2 = input.substr(start);
    return {part1, part2};
}
