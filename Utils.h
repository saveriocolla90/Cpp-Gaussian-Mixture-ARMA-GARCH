#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <numeric>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <float.h>      // DBL_MAX
#include <limits>       // numeric_limits

// Mersenne-Twister  RNG
std::random_device rd;									// Will be used to obtain a seed (by the system entropy) for the random number engine
std::mt19937 gen(rd());									// Seeding the Mersenne-Twister RNG 
std::uniform_real_distribution<> uniform_dist(0., 1.);	// Uniform distribution between [0,1)

/* Data type definition */
typedef std::vector<double> vec_double;
typedef std::vector<vec_double> vecvec_double;
typedef std::array <vecvec_double, 2> matrix;

/* Time series structure */
struct TS
{
    std::vector<uint64_t> time;	// Timestamp (ms)
    vec_double value;	// Values	

    TS& operator*=(const double& _rha) {
        for (auto& val : this->value)
            val *= _rha;
        return *this;
    }

    TS& operator+=(const TS& _rhs) {
        int i = 0;
        for (auto& val : this->value)
            val += _rhs.value.at(i++);
        return *this;
    }
    TS& operator^(const double _pow) {
        int i = 0;
        for (auto& val : this->value)
            val = std::powl(val, _pow);
        return *this;
    }
};
typedef std::vector<TS> vec_TS;

/* ---> General utilities <--- */
template<class T>
void Display(T to_print) {
    std::cout << to_print << std::endl;
};
void Display_TS(TS _in) {
    /* Print a given time series */
    int k = 0;
    Display("Time\tValue\n");
    for (const auto& val : _in.value)
        std::cout << _in.time.at(k++) << "\t" << val << "\n";
    Display("-------------------\n");
    Display("Data: " + std::to_string(_in.value.size()));
};
void WriteTStoFile(TS _in, std::string _filename) {
    /* Write a given time series to file. */

    std::ofstream outfile;
    outfile.open(_filename);

    for (int i = 0; i < _in.time.size(); i++) {
        outfile << _in.time.at(i) << "\t" << _in.value.at(i) << std::endl;
    }
    outfile.close();
};

template<typename T>
bool is_infinite(const T& value)
{
    // Since we're a template, it's wise to use std::numeric_limits<T>
    //
    // Note: std::numeric_limits<T>::min() behaves like DBL_MIN, and is the smallest absolute value possible.
    //

    T max_value = std::numeric_limits<T>::max();
    T min_value = -max_value;

    return !(min_value <= value && value <= max_value);
};

template<typename T>
bool is_nan(const T& value)
{
    // True if NAN
    return value != value;
};

template<typename T>
bool is_valid(const T& value)
{
    return !is_infinite(value) && !is_nan(value);
};


/* ---> Operators overload <--- */
TS operator+(const TS& _lha, const TS& _rha) {
    TS out;
    const TS* ptrmin;
    const TS* ptrmax;
    int offset = _lha.value.size() - _rha.value.size();
    if (offset >= 0) {
        ptrmin = &_rha;
        ptrmax = &_lha;
    }
    else {
        ptrmin = &_lha;
        ptrmax = &_rha;
        offset = std::abs(offset);
    }

    out.time = ptrmin->time;
    out.value.resize(ptrmin->value.size());
    for (int k = 0; k < ptrmin->value.size(); k++) {
        out.value.at(k) = ptrmax->value.at(k + offset) + ptrmin->value.at(k);
    }
    return out;
}
TS operator-(const TS& _lha, const TS& _rha) {
    TS out;  
    const TS* ptrmin;
    const TS* ptrmax;
    int offset = _lha.value.size() - _rha.value.size();
    if (offset >= 0) {
        ptrmin = &_rha;
        ptrmax = &_lha;
    }
    else {
        ptrmin = &_lha;
        ptrmax = &_rha;
        offset = std::abs(offset);
    }

    out.time = ptrmin->time;
    out.value.resize(ptrmin->value.size());
    for (int k = 0; k < ptrmin->value.size(); k++) {
        out.value.at(k) = ptrmax->value.at(k + offset) - ptrmin->value.at(k);
    }
    return out;
}
TS operator*(const TS& _lha, const TS& _rha) {
    TS out;
    const TS* ptrmin;
    const TS* ptrmax;
    int offset = _lha.value.size() - _rha.value.size();
    if (offset >= 0) {
        ptrmin = &_rha;
        ptrmax = &_lha;
    }
    else {
        ptrmin = &_lha;
        ptrmax = &_rha;
        offset = std::abs(offset);
    }

    out.time = ptrmin->time;
    out.value.resize(ptrmin->value.size());
    for (int k = 0; k < ptrmin->value.size(); k++) {
        out.value.at(k) = ptrmax->value.at(k + offset) * ptrmin->value.at(k);
    }
    return out;
}
TS operator/(const TS& _lha, const TS& _rha) {
    TS out;
    const TS* ptrmin;
    const TS* ptrmax;
    int offset = _lha.value.size() - _rha.value.size();
    if (offset >= 0) {
        ptrmin = &_rha;
        ptrmax = &_lha;
    }
    else {
        ptrmin = &_lha;
        ptrmax = &_rha;
        offset = std::abs(offset);
    }

    out.time = ptrmin->time;
    out.value.resize(ptrmin->value.size());
    for (int k = 0; k < ptrmin->value.size(); k++) {
        out.value.at(k) = ptrmax->value.at(k + offset) / ptrmin->value.at(k);
    }
    return out;
}
TS operator*(const double& _lha, const TS& _rha) {
    TS out;
    out.time = _rha.time;
    out.value.resize(_rha.value.size());

    int k = 0;
    for (auto& val : _rha.value) {
        out.value.at(k++) = _lha * val;
    }
    return out;
}
vec_double operator+(const vec_double& _lha, const double& _rha) {
    if (is_valid(_rha)) {
        vec_double out(_lha.size());
        for (int k = 0; k < _lha.size(); k++)
            out.at(k) = _lha.at(k) + _rha;

        return out;
    }
    else return _lha;
};
vec_double operator+(const vec_double& _lha, const vec_double& _rha) {
    vec_double out(_lha.size());
    if (_lha.size() != _rha.size()) {
        std::cout << "Error: Series must be the same size! ";
        throw std::runtime_error("Fatal");
    }
    else {
        for (int k = 0; k < _lha.size(); k++) {
            if (is_valid(_rha.at(k))) {
                out.at(k) = _lha.at(k) + _rha.at(k);
            }
            else {
                out.at(k) = _lha.at(k);
            }
        }
    }
    return out;
};
vec_double operator-(const vec_double& _lha, const double& _rha) {
    if (is_valid(_rha)) {
        vec_double out(_lha.size());
        for (int k = 0; k < _lha.size(); k++)
            out.at(k) = _lha.at(k) - _rha;

        return out;
    }
    else return _lha;
};
vec_double operator-(const vec_double& _lha, const vec_double& _rha) {
    vec_double out(_lha.size());
    if (_lha.size() != _rha.size()) {
        std::cout << "Error: Series must be the same size! ";
        throw std::runtime_error("Fatal");
    }
    else {
        for (int k = 0; k < _lha.size(); k++) {
            if (is_valid(_rha.at(k))) {
                out.at(k) = _lha.at(k) - _rha.at(k);
            }
            else {
                out.at(k) = _lha.at(k);
            }
        }
    }
    return out;
};
vec_double operator*(const vec_double& _lha, const vec_double& _rha) {
    vec_double out(_lha.size());
    if (_lha.size() != _rha.size()) {
        std::cout << "Error: Series must be the same size! ";
        throw std::runtime_error("Fatal");
    }
    else {
        for (int k = 0; k < _lha.size(); k++) {
            if (is_valid(_rha.at(k))) {
                out.at(k) = _lha.at(k) * _rha.at(k);
            }
            else {
                out.at(k) = _lha.at(k);
            }
        }
    }
    return out;
};
vec_double operator/(const vec_double& _lha, const vec_double& _rha) {
    vec_double out(_lha.size());
    if (_lha.size() != _rha.size()) {
        std::cout << "Error: Series must be the same size! ";
        throw std::runtime_error("Fatal");
    }
    else {
        for (int k = 0; k < _lha.size(); k++) {
            if (is_valid(_rha.at(k))) {
                out.at(k) = _lha.at(k) / _rha.at(k);
            }
            else {
                out.at(k) = _lha.at(k);
            }
        }
    }
    return out;
};
vec_double operator*(const double& _lha, const vec_double& _rha) {
    if (is_valid(_lha)) {
        vec_double out(_rha.size());
        int k = 0;
        for (auto val : _rha)
            out.at(k++) = _lha * val;
        return out;
    }
    else return _rha;
};
vec_double operator/(const double& _lha, const vec_double& _rha) {
    if (is_valid(_lha)) {
        vec_double out(_rha.size());
        int k = 0;
        for (auto val : _rha)
            out.at(k++) = _lha / val;
        return out;
    }
    else return _rha;
};
vec_double operator^(const vec_double& _lha, const double& _rha) {
    if (is_valid(_rha)) {
        vec_double out(_lha.size());
        int k = 0;
        for (auto& val : _lha)
            out.at(k++) = std::powl(val, _rha);
        return out;
    }
    else return _lha;
};
vec_double operator>>(const vec_double& _lha, const int& _idx) {
    auto len = _lha.size();
    vec_double out(int(len) - _idx);
    for (int i = _idx; i < int(len); i++)
        out.at(i - _idx) = _lha.at(i);
    return out;
};
vecvec_double operator>>(const vecvec_double& _lha, const int& _idx) {
    int dim = _lha.size();
    int len = _lha.at(0).size();
    vecvec_double out(dim);
    for (int d = 0; d < dim; d++) {
        out.at(d).resize(len - _idx);
        for (int i = _idx; i < len; i++)
            out.at(d).at(i - _idx) = _lha.at(d).at(i);
    }
    return out;
};


/* ---> Enumerators <--- */
enum GarchParams {
    Delta = 0,
    Beta = 1
};
enum ArmaDerivate {
    A = 0,
    B = 1,
};
enum GarchDerivate {
    Gamma = 0,
    Rho = 1,
    Gamma_0 = 2
};


/* ---> Math <--- */
vecvec_double matrProd(vecvec_double& _lhm, vecvec_double& _rhm) {
    /* Compite the matricial product between two matrices. */

    vecvec_double out;
    if (_lhm.size() != _rhm.at(0).size()) {
        Display("Error: Matrix 1 should have a number of rows equals to the number of columns of matrix 2! ");
        throw std::runtime_error("Fatal");
    }
    else {
        vec_double trans;
        trans.resize(_rhm.size());
        out.resize(_lhm.size());
        for (int r = 0; r < _lhm.size(); r++) {
            out.at(r).resize(_lhm.size());
            for (int j = 0; j < _rhm.at(r).size(); j++) {
                for (int c = 0; c < _rhm.size(); c++)
                    trans.at(c) = _rhm.at(c).at(j);
                out.at(r).at(j) = std::inner_product(_lhm.at(r).begin(), _lhm.at(r).end(), trans.begin(), 0.);
            }
        }
    }
    return out;
};
vec_double matrProd(vecvec_double& _lhm, vec_double& _rha) {
    /* Compite the matricial product between a matrix and a vector. */

    vec_double out;
    if (_lhm.at(0).size() != _rha.size()) {
        Display("Error: Vector dimension should be equals to column matrix dimension ! ");
        throw std::runtime_error("Fatal");
    }
    else {
        out.resize(_rha.size());
        for (int r = 0; r < _lhm.size(); r++)
            out.at(r) = std::inner_product(_lhm.at(r).begin(), _lhm.at(r).end(), _rha.begin(), 0.);
    }
    return out;
};
vecvec_double matrInverse(vecvec_double& _matrix) {
    /* Inverse a 2x2 or 3x3 matrix */

    vecvec_double inverse(_matrix.size());
    for (int r = 0; r < _matrix.size(); r++) 
        inverse.at(r).resize(_matrix.size());

    /* 2x2 */
    if (_matrix.size() == 2) {
        
        auto detInv = 1. / (_matrix[0][0] * _matrix[1][1] - _matrix[0][1] * _matrix[1][0]);
        if (detInv == 0 || std::isnan(detInv)) {
            Display("Error: Matrix is NOT invertible!");
            throw std::runtime_error("Fatal");
        }
        else {
            inverse[0][0] = detInv * _matrix[1][1];
            inverse[1][0] = (-1.) * detInv * _matrix[1][0];
            inverse[0][1] = (-1.) * detInv * _matrix[0][1];
            inverse[1][1] = detInv * _matrix[0][0];
        } 
    }

    /* 3x3 */
    if (_matrix.size() == 3) {
        auto detInv = 1. /
            (_matrix[0][0] * _matrix[1][1] * _matrix[2][2] - _matrix[0][0] * _matrix[1][2] * _matrix[2][1]
                - _matrix[0][1] * _matrix[1][0] * _matrix[2][2] + _matrix[0][1] * _matrix[1][2] * _matrix[2][0]
                + _matrix[0][2] * _matrix[1][0] * _matrix[2][1] - _matrix[0][2] * _matrix[1][1] * _matrix[2][0]);

        if (detInv == 0 || std::isnan(detInv)) {
            Display("Matrix is NOT invertible!");
            throw std::runtime_error("Fatal");
        }
        else {
            inverse[0][0] = detInv * (_matrix[1][1] * _matrix[2][2] - _matrix[1][2] * _matrix[2][1]);
            inverse[1][0] = (-1.) * detInv * (_matrix[1][0] * _matrix[2][2] - _matrix[1][2] * _matrix[2][0]);
            inverse[2][0] = detInv * (_matrix[1][0] * _matrix[2][1] - _matrix[1][1] * _matrix[2][0]);
            inverse[0][1] = (-1.) * detInv * (_matrix[0][1] * _matrix[2][2] - _matrix[0][2] * _matrix[2][1]);
            inverse[1][1] = detInv * (_matrix[0][0] * _matrix[2][2] - _matrix[0][2] * _matrix[2][0]);
            inverse[2][1] = (-1.) * detInv * (_matrix[0][0] * _matrix[2][1] - _matrix[0][1] * _matrix[2][0]);
            inverse[0][2] = detInv * (_matrix[0][1] * _matrix[1][2] - _matrix[0][2] * _matrix[1][1]);
            inverse[1][2] = (-1.) * detInv * (_matrix[0][0] * _matrix[1][2] - _matrix[0][2] * _matrix[1][0]);
            inverse[2][2] = detInv * (_matrix[0][0] * _matrix[1][1] - _matrix[0][1] * _matrix[1][0]);
        } 
    }
    return inverse;
};


/* ---> Statistics <--- */
template <typename T>
T normal_pdf(T x, T m, T s)
{
    static const T inv_sqrt_2pi = 0.3989422804014327;
    T a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
};
vec_double normal_pdf(vec_double _series) {
    vec_double out(_series.size());
    int k = 0;
    for (auto& val : _series)
        out.at(k++) = normal_pdf(val, 0., 1.);
    return out;
};
vec_double log(vec_double _series) {
    vec_double out(_series.size());
    int k = 0;
    for (auto& val : _series)
        out.at(k++) = std::log(val);
    return out;

};


