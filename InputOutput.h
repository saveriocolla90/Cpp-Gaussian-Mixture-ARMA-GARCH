#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

#include "Utils.h"
#include "Settings.h"

/* Data loader */
class Loader {
private:
	std::string data_path_;
	TS data_;

public:
	void setPath(std::string _in_file) { data_path_ = _in_file; };
	std::string getPath() { return data_path_; };
	void load();
    TS getData() { return this->data_; };
    void printData();
};

void Loader::load() {
    /* Loading a time series from text file and store the data into 
       a private class structure. */

    std::ifstream file(data_path_);
    if (file.is_open()) {
        std::string line, data;
        while (std::getline(file, line))
        {
            std::stringstream linestream(line);
            std::getline(linestream, data, '\t');
            this->data_.time.push_back(std::stoi(data));
            std::getline(linestream, data, '\t');
            this->data_.value.push_back(std::stod(data)*conversion_factor);
        }
    }
    else Display("Error in opening file!");
};

void Loader::printData() {
    /* Print to screen current stored time series data. */

    if (this->data_.time.size() > 0) {
        for (int i = 0; i < this->data_.time.size(); i++) {
            auto str_out = std::to_string(this->data_.time.at(i)) + "\t" + std::to_string(this->data_.value.at(i));
            Display(str_out);
        }
    }
    else Display("Printing error: Data size is zero !");
    Display("Data size: " + std::to_string(this->data_.time.size()));
};
