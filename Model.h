#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>
#include "Utils.h"

class Model {
protected:
	int length_;
	std::vector<uint64_t> time_;
	vec_double data_;
	vecvec_double derivate_a_;
	vecvec_double derivate_b_;
public:
	Model() {};
	~Model() = default;
	virtual void setData(TS _ts_in) { this->data_ = _ts_in.value; this->time_ = _ts_in.time; };
	virtual void setParams(std::array<int, 2>) = 0;
	virtual void printParams() = 0;
	virtual void rndInit() = 0;
	virtual bool checkStability() = 0;
	virtual void compute() = 0;
	virtual void output(int) = 0;

	/* Output */
	vecvec_double getDerivate_a() {
		return this->derivate_a_;
	}
	vecvec_double getDerivate_b() {
		return this->derivate_b_;
	}

	// Slicing vector
	//std::vector<uint64_t>& operator[](std::tuple<int, int> _start_end) {
	//	auto start = std::get<0>(_start_end);
	//	auto end = std::get<1>(_start_end);
	//	std::vector<uint64_t>::const_iterator first = this->time_.begin() + start;
	//	std::vector<uint64_t>::const_iterator last = this->time_.begin() + start + end;
	//	std::vector<uint64_t> newVec(first, last);
	//	return newVec;
	//}
};