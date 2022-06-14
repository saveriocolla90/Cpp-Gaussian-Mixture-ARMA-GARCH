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

class GarchModel : public Model {
protected:
	int size_;
	std::array<int, 2> rank_{ 1,1 };	// Q, P
	double d0_, gamma0_;
	vec_double delta_, beta_;
	vec_double gamma_, rho_;
	vec_double var_;
	vec_double var_hat_;
	vec_double derivate_gamma0_;
	vecvec_double derivate_gamma_;
	vecvec_double derivate_rho_;
public:
	GarchModel(std::array<int, 2> _rank, int _size) : Model() {
		this->size_ = _size;						// (N - R - Q) data lenght 
		this->var_.resize(this->size_);
		this->derivate_b_.resize(_rank.at(0));		// R components (needs to fix)
		this->derivate_a_.resize(_rank.at(1));		// S components (needs to fix)
		this->derivate_gamma_.resize(_rank.at(0));	// Q components
		this->derivate_rho_.resize(_rank.at(1));	// P components

		// Resize the derivatives vectors
		int P = _rank.at(1);
		this->derivate_gamma0_.resize(this->size_);
		for (auto& da : derivate_a_)
			da.resize(this->size_);
		for (auto& db : derivate_b_)
			db.resize(this->size_);
		for (auto& dg : derivate_gamma_)
			dg.resize(this->size_);
		for (auto& dr : derivate_rho_)
			dr.resize(this->size_ - P);

		this->setParams(_rank);
		this->rndInit();
		this->transform(false);
	};

	/* Preliminary */
	void transform(bool _backTo) {
		int k = 0;
		if (_backTo) {
			d0_ = std::exp(gamma0_);
			for (auto g : gamma_)
				delta_.at(k++) = std::exp(g);
			k = 0;
			for (auto r : rho_)
				beta_.at(k++) = std::exp(r);
		}
		else {
			gamma0_ = std::log(d0_);
			for (auto d : delta_)
				gamma_.at(k++) = std::log(d);
			k = 0;
			for (auto b : beta_)
				rho_.at(k++) = std::log(b);

		}
	}
	void setParams(std::array<int, 2> _rank) override {
		rank_ = _rank;
		d0_ = uniform_dist(gen);
		for (int q = 0; q < rank_[0]; q++) {
			delta_.push_back(uniform_dist(gen));
			gamma_.push_back(0.);
		}

		for (int p = 0; p < rank_[1]; p++) {
			beta_.push_back(uniform_dist(gen));
			rho_.push_back(0.);
		}
	};
	void resetParmas(vec_double _delta, vec_double _beta) {
		this->delta_ = _delta;
		this->beta_ = _beta;
	}
	void rndInit() override {
		for (int q = 0; q < rank_[0]; q++)
			delta_.at(q) = uniform_dist(gen);
		for (int p = 0; p < rank_[1]; p++)
			beta_.at(p) = uniform_dist(gen);
		d0_ = uniform_dist(gen);
	};
	bool checkStability() override {
		double sum{ 0. };
		sum = std::accumulate(this->delta_.begin(), this->delta_.end(), sum);
		sum = std::accumulate(this->beta_.begin(), this->beta_.end(), sum);
		return sum < 1;
	};

	/* Input */
	void incrParam(vec_double& incr, GarchDerivate p) {
		/* Increment the given parameter by input values. */

		if (p == GarchDerivate::Gamma)
			this->gamma_ = this->gamma_ - incr;

		if (p == GarchDerivate::Rho)
			this->rho_ = this->rho_ - incr;
	}
	void incrParam(double& incr, GarchDerivate p) {
		/* Increment the given parameter by input values. */

		if (is_valid(incr)) {
			if (p == GarchDerivate::Gamma_0)
				this->gamma0_ = this->gamma0_ - incr;
		}
	}

	/* Outout */
	std::array<int, 2> getRank() { return rank_; };
	TS getModelSeries() {
		TS out;
		int offset = this->rank_.at(0);
		out.time.resize(this->size_ - offset);
		for (int k = 0; k < this->size_ - offset; k++) {
			out.time.at(k) = this->time_.at(k + 2.*offset);
		}
		out.value = this->var_hat_;
		return out;
	}
	TS getSeries() {
		TS out;
		out.time.resize(this->size_);
		int offset = this->rank_.at(0);
		for (int k = 0; k < this->size_; k++) {
			out.time.at(k) = this->time_.at(k + offset);
		}
		out.value = this->var_;
		return out;
	}
	vec_double getVariance() {
		return this->var_;
	}
	vec_double getDerivate_gamma0() {
		return this->derivate_gamma0_;
	}
	vecvec_double getDerivate_gamma() {
		return this->derivate_gamma_;
	}
	vecvec_double getDerivate_rho() {
		return this->derivate_rho_;
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
		else if (_idx == 2) {
			out.resize(this->rank_.at(0));
			int k = 0;
			for (auto& dev : derivate_gamma_) {
				out.at(k).time = this->time_;
				out.at(k).value = this->derivate_gamma_.at(k);
				k++;
			}
		}
		else if (_idx == 3) {
			out.resize(this->rank_.at(1));
			int k = 0;
			for (auto& dev : derivate_rho_) {
				out.at(k).time = this->time_;
				out.at(k).value = this->derivate_rho_.at(k);
				k++;
			}
		}
		return out;
	}
	vec_double getParams(GarchParams p) {
		if (p == GarchParams::Delta)
			return this->delta_;
		if (p == GarchParams::Beta)
			return this->beta_;
	}
	void output(int _rank) override {
		auto write_out = this->getSeries();
		std::ofstream outfile;
		outfile.open(file_variances + "_" + std::to_string(_rank) + extension);

		int k = 0;
		for (const auto& val : write_out.value) {
			outfile << write_out.time.at(k++) << "\t" << val << std::endl;
		}
		outfile.close();
	}

	/* Expected values */
	void computeExpected() {
		/* Compute the current expected model's values. */
		int Q = rank_.at(0);
		int P = rank_.at(1);
		std::vector<double> transpost;

		this->var_hat_.resize(size_ - P);
		for (int t = Q + P; t < size_ + Q; t++) {
			// GAR part
			transpost.clear();
			for (int q = 1; q <= Q; q++)
				transpost.push_back(std::powl(this->data_.at(t - q), 2.));
			this->var_hat_.at(t - Q - P) = this->d0_ + std::inner_product(delta_.begin(), delta_.end(), transpost.begin(), 0.);

			// CH part
			transpost.clear();
			for (int p = 1; p <= P; p++)
				transpost.push_back(this->var_.at(t - Q - p));
			this->var_hat_.at(t - Q - P) += std::inner_product(beta_.begin(), beta_.end(), transpost.begin(), 0.);
		}
	}

	/* Variance */
	void compute() override {
		/* Compute the current residual model's part. */
		int Q = rank_.at(0);
		int P = rank_.at(1);
		std::vector<double> transpost;

		for (int t = Q; t < size_ + Q; t++) {
			// GAR part	
			transpost.clear();
			for (int i = 1; i <= Q; i++)
				transpost.push_back(std::powl(this->data_.at(t - i), 2.));
			this->var_.at(t - Q) = this->d0_ + std::inner_product(delta_.begin(), delta_.end(), transpost.begin(), 0.);

			// CH part
			if (t >= Q + P) {
				transpost.clear();
				for (int i = 1; i <= P; i++)
					transpost.push_back(this->var_.at(t - Q - i));
				this->var_.at(t - Q) += std::inner_product(beta_.begin(), beta_.end(), transpost.begin(), 0.);
			}
		}
	};

	/* Derivatives */
	void derivate(vecvec_double& _ts_in_a, vecvec_double& _ts_in_b, vec_double& _resid) {
		/* Derivate variances respect to ARMA parameters */
		int Q = rank_.at(0);
		int P = rank_.at(1);
		vec_double transpost, resid_vec, coefs;

		/* ----> Derivate variances respect to ARMA parameters <---- */
		// Respect B
		for (int r = 0; r < _ts_in_b.size(); r++) {
			for (int t = Q; t < size_ + Q; t++) {
				transpost.clear();
				resid_vec.clear();
				for (int i = 1; i <= Q; i++) {
					transpost.push_back(std::exp(gamma_.at(i - 1)) * _ts_in_b.at(r).at(t - i));
					resid_vec.push_back(_resid.at(t - i));
				}
				this->derivate_b_.at(r).at(t - Q) = 2. * std::inner_product(resid_vec.begin(), resid_vec.end(), transpost.begin(), 0.);

				// Regressive part
				if (t >= Q + P) {
					transpost.clear();
					coefs.clear();
					for (int i = 1; i <= P; i++) {
						transpost.push_back(this->derivate_b_.at(r).at(t - Q - i));
						coefs.push_back(std::exp(rho_.at(i - 1)));
					}
					this->derivate_b_.at(r).at(t - Q) += std::inner_product(coefs.begin(), coefs.end(), transpost.begin(), 0.);
				}
			}
		}

		// Respect A
		for (int s = 0; s < _ts_in_a.size(); s++) {
			for (int t = Q; t < size_ + Q; t++) {
				transpost.clear();
				resid_vec.clear();
				for (int i = 1; i <= Q; i++) {
					transpost.push_back(std::exp(gamma_.at(i - 1)) * _ts_in_a.at(s).at(t - i));
					resid_vec.push_back(_resid.at(t - i));
				}
				this->derivate_a_.at(s).at(t - Q) = 2. * std::inner_product(resid_vec.begin(), resid_vec.end(), transpost.begin(), 0.);

				// Regressive part
				if (t >= Q + P) {
					transpost.clear();
					coefs.clear();
					for (int i = 1; i <= P; i++) {
						transpost.push_back(this->derivate_a_.at(s).at(t - Q - i));
						coefs.push_back(std::exp(rho_.at(i - 1)));
					}
					this->derivate_a_.at(s).at(t - Q) += std::inner_product(coefs.begin(), coefs.end(), transpost.begin(), 0.);
				}
			}
		}


		/* ----> Derivate variances respect to GARCH parameters <---- */
		// Respect GAMMA_0
		for (int t = Q; t < size_ + Q; t++) {
			this->derivate_gamma0_.at(t - Q) = std::exp(gamma0_);

			if (t >= Q + P) {
				transpost.clear();
				coefs.clear();
				for (int i = 1; i <= P; i++) {
					transpost.push_back(this->derivate_gamma0_.at(t - Q - i));
					coefs.push_back(std::exp(rho_.at(i - 1)));
				}
				this->derivate_gamma0_.at(t - Q) += std::inner_product(coefs.begin(), coefs.end(), transpost.begin(), 0.);
			}
		}

		// Respect GAMMA
		for (int q = 0; q < Q; q++) {
			for (int t = Q; t < size_ + Q; t++) {
				this->derivate_gamma_.at(q).at(t - Q) = std::powl(_resid.at(t - q - 1), 2.) * std::exp(gamma_.at(q));

				if (t >= Q + P) {
					transpost.clear();
					coefs.clear();
					for (int i = 1; i <= P; i++) {
						transpost.push_back(this->derivate_gamma_.at(q).at(t - Q - i));
						coefs.push_back(std::exp(rho_.at(i - 1)));
					}
					this->derivate_gamma_.at(q).at(t - Q) += std::inner_product(coefs.begin(), coefs.end(), transpost.begin(), 0.);
				}
			}
		}

		// Respect RHO
		for (int p = 0; p < P; p++) {
			for (int t = Q; t < size_; t++) {
				this->derivate_rho_.at(p).at(t - Q) = this->var_.at(t - p - 1) * std::exp(rho_.at(p));

				if (t >= Q + P) {
					transpost.clear();
					coefs.clear();
					for (int i = 1; i <= P; i++) {
						transpost.push_back(this->derivate_rho_.at(p).at(t - Q - i));
						coefs.push_back(std::exp(rho_.at(i - 1)));
					}
					this->derivate_rho_.at(p).at(t - Q) += std::inner_product(coefs.begin(), coefs.end(), transpost.begin(), 0.);
				}
			}
		}
	}

	/* Display functions */
	void printParams() {
		/* Display all the curent parameters value. */
		Display("Parameter d0: " + std::to_string(this->d0_));
		Display("\nParameter DELTA:\nRank:" + std::to_string(this->rank_.at(0)));
		for (const auto& d : this->delta_)
			Display(std::to_string(d) + "\t");

		Display("\nParameter BETA:\nRank:" + std::to_string(this->rank_.at(1)));
		for (const auto& b : this->beta_)
			Display(std::to_string(b) + "\t");
	}
};