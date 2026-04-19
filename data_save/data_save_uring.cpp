#include "data_save_uring.h"

static void die(const string &msg)
{
    fprintf(stderr, "FATAL: %s (errno=%d, %s)\n", msg.c_str(), errno, strerror(errno));
    exit(1);
}

static string make_filename(const string &dir, const string &prefix, double index)
{
    string index_str = to_string(index);
    size_t dot_pos = index_str.find('.');
    if (dot_pos != string::npos)
    {
        // Keep 2 digits after the decimal point
        index_str = index_str.substr(0, dot_pos + 3);
    }
    else
    {
        // If no decimal point, add .00
        index_str += ".00";
    }
    return dir + "/" + prefix + "_DM" + index_str + ".dat";
}

DataSave_Uring::DataSave_Uring(string filename, int numDMs)
{
}
