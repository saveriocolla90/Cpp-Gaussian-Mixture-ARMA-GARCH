#pragma once

#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <math.h>
#include "Utils.h"
#include "Model.h"
#include "Settings.h"


class ArmaModel : public Model {
protected:
	int size_;
	std::array<int, 2> rank_{ 1,1 };	// R, S
	vec_double a_, b_;
	vec_double resid_;
	vec_double y_hat_;
public:
	ArmaModel(std::array<int, 2> _rank, int _size) : Model() {
		this->size_ = _size;					// (N - R) data lenght
		this->resid_.resize(this->size_);
		this->derivate_b_.resize(_rank.at(0));	// R components
		this->derivate_a_.resize(_rank.at(1));	// S components

		// Resize the derivatives vectors
		for (auto& da : derivate_a_)
			da.resize(this->size_);
		for (auto& db : derivate_b_)
			db.resize(this->size_);

		this->setParams(_rank);
	};

	/* Input */
	void incrParam(vec_double& incr, ArmaDerivate p) {
		/* Increment the given parameter by input values. */

		if (p == ArmaDerivate::A)
			this->a_ = this->a_ + incr;

		if (p == ArmaDerivate::B)
			this->b_ = this->b_ + incr;
	}

	/* Outout */
	TS getModelSeries() {
		TS out;
		int offset = this->rank_.at(0);
		out.time.resize(this->size_ - offset);
		for (int k = 0; k < this->size_ - offset; k++) {
			out.time.at(k) = this->time_.at(k + 2.*offset);
		}
		out.value = this->y_hat_;
		return out;
	}
	TS getSeries() {
		TS out;
		out.time.resize(this->size_);
		int offset = this->rank_.at(0);
		for (int k = 0; k < this->size_; k++) {
			out.time.at(k) = this->time_.at(k + offset);
		}
		out.value = this->resid_;
		return out;
	}
	vec_TS getVecSeries(int _idx) {
		vec_TS out;
		if (_idx == 0) {
			out.resize(this->rank_.at(1));
			int k = 0;
			for (auto& dev : derivate_a_) {
				out.at(k).time = this->time_;
				out.at(k).value = this->derivate_a_.at(k);
				k++;
			}
		}
		else if (_idx == 1) {
			out.resize(this->rank_.at(0));
			int k = 0;
			for (auto& dev : derivate_b_) {
				out.at(k).time = this->time_;
				out.at(k).value = this->derivate_b_.at(k);
				k++;
			}
		}
		return out;
	}
	vec_double getResiduals() { return this->resid_; }
	std::array<int, 2> getRank() { return rank_; };
	void output(int _rank) override {
		auto write_out = this->getSeries();
		std::ofstream outfile;
		outfile.open(file_residulas + "_" + std::to_string(_rank) + extension);

		int k = 0;
		for (const auto& val : write_out.value) {
			outfile << write_out.time.at(k++) << "\t" << val << std::endl;
		}
		outfile.close();
	};

	/* Preliminary */
	void setParams(std::array<int, 2> _rank) override {
		rank_ = _rank;
		for (int r = 0; r < rank_.at(0); r++)
			b_.push_back(uniform_dist(gen));
		for (int s = 0; s < rank_.at(1); s++)
			a_.push_back(uniform_dist(gen));
	}
	void rndInit() override {
		for (int r = 0; r < rank_.at(0); r++)
			b_.at(r) = uniform_dist(gen);
		for (int s = 0; s < rank_.at(1); s++)
			a_.at(s) = uniform_dist(gen);
	}
	bool checkStability() override {
		return true;
	}

	/* Expected values */
	void computeExpected() {
		/* Compute the current expected model's values. */
		int R = rank_.at(0);
		int S = rank_.at(1);
		std::vector<double> transpost;

		this->y_hat_.resize(size_ - S);
		for (int t = R + S; t < size_ + R; t++) {
			// AR part
			transpost.clear();
			for (int r = 1; r <= R; r++)
				transpost.push_back(this->data_.at(t - r));
			this->y_hat_.at(t - R - S) = std::inner_product(b_.begin(), b_.end(), transpost.begin(), 0.);

			// MA part
			transpost.clear();
			for (int s = 1; s <= S; s++)
				transpost.push_back(this->resid_.at(t - R - s));
			this->y_hat_.at(t - R - S) += std::inner_product(a_.begin(), a_.end(), transpost.begin(), 0.);
		}
	}

	/* Residual */
	void compute() override {
		/* Compute the current residual model's part. */
		int R = rank_.at(0);
		int S = rank_.at(1);
		std::vector<double> transpost;

		for (int t = R; t < size_ + R; t++) {
			// AR part	
			transpost.clear();
			for (int i = 1; i <= R; i++)
				transpost.push_back(this->data_.at(t - i));
			this->resid_.at(t - R) = this->data_.at(t) - std::inner_product(b_.begin(), b_.end(), transpost.begin(), 0.);

			// MA part
			if (t >= R + S) {
				transpost.clear();
				for (int i = 1; i <= S; i++)
					transpost.push_back(this->resid_.at(t - R - i));
				this->resid_.at(t - R) -= std::inner_product(a_.begin(), a_.end(), transpost.begin(), 0.);
			}
		}
	}

	/* Derivatives */
	void derivate() {
		/* Derivate residuals respect to ARMA parameters */
		int R = rank_.at(0);
		int S = rank_.at(1);
		std::vector<double> transpost;

		// Respect B
		for (int r = 0; r < R; r++) {
			for (int t = R; t < size_ + R; t++) {
				this->derivate_b_.at(r).at(t - R) = (-1.) * this->data_.at(t - r - 1);

				// Regressive part
				if (t >= R + S) {
					transpost.clear();
					for (int i = 1; i <= S; i++)
						transpost.push_back(this->derivate_b_.at(r).at(t - R - i));
					this->derivate_b_.at(r).at(t - R) -= std::inner_product(a_.begin(), a_.end(), transpost.begin(), 0.);
				}
			}
		}

		// Respect A
		for (int s = 0; s < S; s++) {
			for (int t = S; t < size_; t++) {
				this->derivate_a_.at(s).at(t - S) = (-1.) * this->resid_.at(t - s - 1);

				// MA part
				if (t >= 2 * S) {
					transpost.clear();
					for (int i = 1; i <= S; i++)
						transpost.push_back(this->derivate_a_.at(s).at(t - S - i));
					this->derivate_a_.at(s).at(t - S) -= std::inner_product(a_.begin(), a_.end(), transpost.begin(), 0.);
				}
			}
		}
	}

	/* Display functions */
	void printParams() {
		/* Display all the curent parameters value. */
		Display("Parameter B:\nRank:" + std::to_string(this->rank_.at(0)));
		for (const auto& b : this->b_)
			Display(std::to_string(b) + "\t");

		Display("\n\nParameter A:\nRank:" + std::to_string(this->rank_.at(1)));
		for (const auto& a : this->a_)
			Display(std::to_string(a) + "\t");
	}
};