#ifndef GPR_H
#define GPR_H


#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <math.h>

//#ifdef USE_DOUBLE_PRECISION
//typedef double REALTYPE;
//#else
//typedef float REALTYPE;
//#endif



template<typename REALTYPE>
class GaussianProcessRegression{

  typedef Eigen::Matrix<REALTYPE,Eigen::Dynamic,Eigen::Dynamic> MatrixXr;
  typedef Eigen::Matrix<REALTYPE,Eigen::Dynamic,1> VectorXr;

  MatrixXr input_data_;
  MatrixXr output_data_;
  MatrixXr KXX;
  MatrixXr KXX_, KXXBeta_;
  VectorXr KXx;
  //MatrixXr KxX;
  VectorXr Kxx; // New, for variance calculation

  int n_data_;
  bool b_need_prepare_;

  double l_scale_, l_scale2_;
  double sigma_f_;
  double sigma_n_;

  // For NN kernel
  double betaNN_;

  // For RQ kernel
  double alphaRQ_;

  // For HY kernel
  double sigma_fHY_;
  double sigma_nHY_;
  double aHY_;
  double bHY_;

  // Kernel chooser
  std::string kernel_;

  // For constant kernel
  double cConst_;

  VectorXr dist;

  VectorXr regressors;

  //  std::vector<Eigen::FullPivLU<MatrixXr> > decompositions_;
  MatrixXr alpha_;
  MatrixXr beta_; // New (variance)

  VectorXr var_;
  bool exists_variance_;
  
public:
  GaussianProcessRegression(int inputDim, int outputDim);

  void SetHyperParams(double l, double f, double n){l_scale_ = l; sigma_f_ = f; sigma_n_ = n;};
  void GetHyperParams(double & l, double & f, double & n){l = l_scale_; f = sigma_f_; n = sigma_n_;};

  void SetHyperParamsNN(double l, double f, double n, double b){l_scale_ = l; sigma_f_ = f; sigma_n_ = n; betaNN_ = b;};
  void GetHyperParamsNN(double & l, double & f, double & n, double & b){l = l_scale_; f = sigma_f_; n = sigma_n_; b = betaNN_;};

  void SetHyperParamsRQ(double l, double f, double n, double a){l_scale_ = l; sigma_f_ = f; sigma_n_ = n; alphaRQ_ = a;};
  void GetHyperParamsRQ(double & l, double & f, double & n, double & a){l = l_scale_; f = sigma_f_; n = sigma_n_; a = alphaRQ_;};

  void SetHyperParamsRQNN(double l, double f, double n, double b, double a){l_scale_ = l; sigma_f_ = f; sigma_n_ = n; betaNN_ = b; alphaRQ_ = a;};
  void GetHyperParamsRQNN(double & l, double & f, double & n, double & b, double & a){l = l_scale_; f = sigma_f_; n = sigma_n_; b = betaNN_; a = alphaRQ_;};

  void SetHyperParamsAll(double l, double l2, double f, double n, double b, double a, double c, double ahy, double bhy, std::string kernel)
          {l_scale_ = l; l_scale2_ = l2; sigma_f_ = f; sigma_n_ = n; betaNN_ = b; alphaRQ_ = a; cConst_ = c; aHY_ = ahy; bHY_ = bhy; kernel_ = kernel;};
  void GetHyperParamsAll(double & l, double & l2, double & f, double & n, double & b, double & a, double & c, double & ahy, double & bhy, std::string & kernel)
          {l = l_scale_; l2 = l_scale2_; f = sigma_f_; n = sigma_n_; b = betaNN_; a = alphaRQ_; c = cConst_; ahy = aHY_; bhy = bHY_; kernel = kernel_;};

  void SetKernel(std::string k){kernel_ = k;};
  void GetKernel(std::string& k){k = kernel_;};

  // add data one by one
  void AddTrainingData(const VectorXr& newInput, const VectorXr& newOutput);
  // batch add data
  void AddTrainingDataBatch(const MatrixXr& newInput,const MatrixXr& newOutput);

  REALTYPE SQEcovFuncD(VectorXr x1,VectorXr x2);
  void Debug();

  MatrixXr SQEcovFunc(MatrixXr x1);
  VectorXr SQEcovFunc(MatrixXr x1, VectorXr x2);

  REALTYPE NNCovFuncD(VectorXr x1, VectorXr x2);
  MatrixXr NNCovFunc(MatrixXr x1);
  VectorXr NNCovFunc(MatrixXr x1, VectorXr x2);

  REALTYPE OUCovFuncD(VectorXr x1, VectorXr x2);
  MatrixXr OUCovFunc(MatrixXr x1);
  VectorXr OUCovFunc(MatrixXr x1, VectorXr x2);

  REALTYPE RQCovFuncD(VectorXr x1, VectorXr x2);
  MatrixXr RQCovFunc(MatrixXr x1);
  VectorXr RQCovFunc(MatrixXr x1, VectorXr x2);

  REALTYPE HYCovFuncD(VectorXr x1, VectorXr x2);

  REALTYPE CCovFuncD(VectorXr x1, VectorXr x2);

  REALTYPE OUSQECovFuncD(VectorXr x1, VectorXr x2);


  // these are fast methods 
  void PrepareRegression(bool force_prepare = false);
  VectorXr DoRegression(const VectorXr & inp,bool prepare = false);
  VectorXr DoRegressionVariance(const VectorXr & inp,bool prepare = false);
  VectorXr DoRegressionNNVariance(const VectorXr & inp,bool prepare = false);
  VectorXr GetVariance();

  // NN version
  void PrepareRegressionNN(bool force_prepare = false);
  VectorXr DoRegressionNN(const VectorXr & inp,bool prepare = false);

  //VectorXr getRegressionVariance(const VectorXr & inp,bool prepare = false); // Was new, but not needed

  // these are the old implementations that are slow, inaccurate and easy to understand
  void PrepareRegressionOld(bool force_prepare = false);
  VectorXr DoRegressionOld(const VectorXr & inp,bool prepare = false);

  int get_n_data(){return n_data_;};
  const MatrixXr& get_input_data()  {return input_data_;};
  const MatrixXr& get_output_data()  {return output_data_;};

  void ClearTrainingData();
};

#include "gaussian_process_regression.hxx"
#endif //GPR_H
