#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm> // For std::remove
#include <utility> // For std:pair
#include <sstream>

#include "globals.h"

auto constexpr blockSize = 4096;
auto constexpr frameSamples = 2048;

using namespace std;

class PSRDADA
{
public:
    PSRDADA(string filename);
    ~PSRDADA();
    void initBuffer(size_t n);
    void readSamples();

private:
    pair<string, string> splitByMultipleSpaces(const string& input);

public:
    char headerBuffer[blockSize];
    ifstream infile;
    string infileName;
    unsigned long filesize;
    unsigned long readCount;
    string outfileName;
    float bw;
    float fc;
    int nbit;
    int nchan;
    int ndim;
    int npol;
    int hdr_size;
    uint16_pair *pol1_in;
    uint16_pair *pol2_in;
    int frameCount;
};