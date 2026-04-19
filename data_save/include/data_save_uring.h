#pragma once
#include "globals.h"
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>

using namespace std;

class DataSave_Uring
{
public:
    DataSave_Uring(string filename, int numDMs);
};