#ifndef GAUSSIAN_PROCESS_REGRESSION_HXX
#define GAUSSIAN_PROCESS_REGRESSION_HXX


#include "gaussian_process_regression/gaussian_process_regression.h"

template<typename R>
GaussianProcessRegression<R>::GaussianProcessRegression(int inputDim,int outputDim)
{
  input_data_.resize(inputDim,0);
  output_data_.resize(outputDim,0);
  n_data_ = 0;
  exists_variance_ = false;
}



template<typename R>
void GaussianProcessRegression<R>::AddTrainingData(const VectorXr& newInput,const VectorXr& newOutput)
{
  n_data_++;
  if(n_data_>=input_data_.cols()){
    input_data_.conservativeResize(input_data_.rows(),n_data_);
    output_data_.conservativeResize(output_data_.rows(),n_data_);
  }
  input_data_.col(n_data_-1) = newInput;
  output_data_.col(n_data_-1) = newOutput;
  b_need_prepare_ = true;
}

// void show_dim(Eigen::MatrixXf a){
//   std::cout<<a.rows()<<" "<<a.cols()<<std::endl;
// }

template<typename R>
void GaussianProcessRegression<R>::AddTrainingDataBatch(const MatrixXr& newInput, const MatrixXr& newOutput)
{
  // sanity check of provided data
  assert(newInput.cols() == newOutput.cols());
  // if this is the first data, just add it..
  if(n_data_ == 0){
    input_data_ = newInput;
    output_data_ = newOutput;
    n_data_ = input_data_.cols();
  }
  // if we already have data, first check dimensionaly match
  else{
    assert(input_data_.rows() == newInput.rows());
    assert(output_data_.rows() == newOutput.rows());
    size_t n_data_old = n_data_;
    n_data_ += newInput.cols();
    // resize the matrices
    if(n_data_ > input_data_.cols()){
      input_data_.conservativeResize(input_data_.rows(),n_data_);
      output_data_.conservativeResize(output_data_.rows(),n_data_);
    }
    // insert the new data using block operations
    input_data_.block(0,n_data_old,newInput.rows(),newInput.cols()) = newInput;
    output_data_.block(0,n_data_old,newOutput.rows(),newOutput.cols()) = newOutput;
  }
  // in any case after adding a batch of data we need to recompute decomposition (in lieu of matrix inversion)
  b_need_prepare_ = true;
}


template<typename R>
R GaussianProcessRegression<R>::SQEcovFuncD(VectorXr x1, VectorXr x2)
{
  dist = x1-x2;
  double d = dist.dot(dist);
  d = sigma_f_*sigma_f_*exp(-1/l_scale_/l_scale_/2*d);
  return d;
}

template<typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::SQEcovFunc(MatrixXr x1, VectorXr x2){
  int nCol = x1.cols();
  VectorXr KXx(nCol);
  for(int i=0;i<nCol;i++){
    KXx(i)=SQEcovFuncD(x1.col(i),x2);
  }
  return KXx;
}

template<typename R>
R GaussianProcessRegression<R>::NNCovFuncD(VectorXr x1, VectorXr x2)
//template<typename R>
//R GaussianProcessRegression<R>::NNcovFuncD()
{
  double d = x1.dot(x2)/l_scale_/l_scale_ + betaNN_;
  double d1 = x1.dot(x1)/l_scale_/l_scale_ + betaNN_ + 1.0;
  double d2 = x2.dot(x2)/l_scale_/l_scale_ + betaNN_ + 1.0;
  
  double nn = sigma_f_*sigma_f_*asin(d/sqrt(d1*d2));
  return nn;
}

template<typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::NNCovFunc(MatrixXr x1, VectorXr x2){
  
  int nCol = x1.cols();
  VectorXr KXx(nCol);
  for(int i=0;i<nCol;i++){
  	if (kernel_ == "nn") KXx(i)=NNCovFuncD(x1.col(i),x2);
  	if (kernel_ == "c") KXx(i)=CCovFuncD(x1.col(i),x2);
  	if (kernel_ == "hy") KXx(i)=HYCovFuncD(x1.col(i),x2);
  	if (kernel_ == "ou") KXx(i)=OUCovFuncD(x1.col(i),x2);
  	if (kernel_ == "sqe") KXx(i)=SQEcovFuncD(x1.col(i),x2);
  	if (kernel_ == "rq") KXx(i)=RQCovFuncD(x1.col(i),x2);
  	if (kernel_ == "ousqe") KXx(i)=OUSQECovFuncD(x1.col(i),x2);
  }
  return KXx;
}

template<typename R>
R GaussianProcessRegression<R>::OUCovFuncD(VectorXr x1, VectorXr x2)
//template<typename R>
//R GaussianProcessRegression<R>::NNcovFuncD()
{  
  double d = (x1-x2).cwiseAbs().sum();
  //double d = 0.0;
  //for (auto& n : dist) {
  //	d += n.norm();
  //}
  //double d = dist.lpNorm<1>();
  d = sigma_f_*sigma_f_*exp(-1*d/l_scale_);
  return d;
  //return x1.dot(x2);
}

template<typename R>
R GaussianProcessRegression<R>::OUSQECovFuncD(VectorXr x1, VectorXr x2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                )
//template<typename R>
//R GaussianProcessRegression<R>::NNcovFuncD()
{  
  double d = (x1-x2).cwiseAbs().sum();
  auto diff = x1-x2;
  double d2 = diff.dot(diff); // Testing if this makes a difference for dropoff velocity!
  //double e1 = exp(-(1*d/l_scale_));
  //double e2 = exp(-(1*d2/l_scale2_/l_scale2_/2));
  //std::cout << "overflowing: d: " << d << " d2: " << d2 << " e1: " << e1 << " e2: " << e2 << std::endl;
  double ein1 = exp(-(1*d/l_scale_));
  double ein2 = exp(-(1*d2/l_scale2_/l_scale2_/2));
  
  //std::cout << "ein1: " << ein1 << " ein2: " << ein2 << std::endl; 

  double argument = 0.06 * ein1 + 0.94 * ein2;
  //double distance = sqrt(d2);
  //if (distance <= 0.25) argument = fmax((1.0 - distance / 0.15), 0.0) * ein1 + fmin((distance / 0.15),1.0) * ein2;
  //else argument = ein2;

  return sigma_f_*sigma_f_*argument;
  //return sigma_f_*sigma_f_*exp(-(1*d/l_scale_) - (1/l_scale2_/l_scale2_/2*d2));
  //return x1.dot(x2);

}

//template<typename R>
//R GaussianProcessRegression<R>::OUSQECovFuncD(VectorXr x1, VectorXr x2)
////template<typename R>
////R GaussianProcessRegression<R>::NNcovFuncD()
//{  
//  double d = (x1-x2).cwiseAbs().sum();
//  //double d = 0.0;
//  //for (auto& n : dist) {
//  //	d += n.norm();
//  //}
//  //double d = dist.lpNorm<1>();
//  d = sigma_f_*sigma_f_*exp(-1*d/l_scale_);
//
//  dist = x1-x2;
//  double d2 = dist.dot(dist);
//  d2 = sigma_f_*sigma_f_*exp(-1/l_scale_/l_scale_/2*d2);


//  return d+d2;
//  //return x1.dot(x2);
//}

template<typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::OUCovFunc(MatrixXr x1, VectorXr x2){
  
  int nCol = x1.cols();
  VectorXr KXx(nCol);
  for(int i=0;i<nCol;i++){
    KXx(i)=OUCovFuncD(x1.col(i),x2);
  }
  return KXx;
}

template<typename R>
R GaussianProcessRegression<R>::HYCovFuncD(VectorXr x1, VectorXr x2)
//template<typename R>
//R GaussianProcessRegression<R>::NNcovFuncD()
{  
  //dist = x1-x2;
  //double d = dist.dot(dist);
  //VectorXr res = min(x1, x2);
  double d = x1.dot(x2);
  double nn = sigma_f_*sigma_f_*tanh(aHY_*d+bHY_);

  return nn;
}

template<typename R>
R GaussianProcessRegression<R>::CCovFuncD(VectorXr x1, VectorXr x2)
//template<typename R>
//R GaussianProcessRegression<R>::NNcovFuncD()
{  
  return cConst_;
}

template<typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::RQCovFunc(MatrixXr x1, VectorXr x2){
  
  int nCol = x1.cols();
  VectorXr KXx(nCol);
  for(int i=0;i<nCol;i++){
    KXx(i)=RQCovFuncD(x1.col(i),x2);
  }
  return KXx;
}

template<typename R>
R GaussianProcessRegression<R>::RQCovFuncD(VectorXr x1, VectorXr x2)
//template<typename R>
//R GaussianProcessRegression<R>::NNcovFuncD()
{  
  dist = (x1-x2);
  double nn = sigma_f_*sigma_f_*pow(1+(dist.dot(dist)/(2.0*alphaRQ_*pow(l_scale_,2.0))),-alphaRQ_);

  return nn;
}


template <typename R>
void GaussianProcessRegression<R>::PrepareRegression(bool force_prepare){
  if(!b_need_prepare_ & !force_prepare)
    return;

  KXX = SQEcovFunc(input_data_);
  KXX_ = KXX;
  // add measurement noise
  for(int i=0;i<KXX.cols();i++)
    KXX_(i,i) += sigma_n_*sigma_n_;
  alpha_.resize(output_data_.rows(),output_data_.cols());
  beta_.resize(output_data_.rows(),output_data_.cols()); // New
  // pretty slow decomposition to compute
  //Eigen::FullPivLU<MatrixXr> decomposition(KXX_);
  // this is much much faster:
  Eigen::LDLT<MatrixXr> decomposition(KXX_);
  for (size_t i=0; i < output_data_.rows(); ++i)
    {
      alpha_.row(i) = (decomposition.solve(output_data_.row(i).transpose())).transpose();
      //beta_.row(i) = (decomposition.solve(decomposition.row(i).transpose())).transpose();
    }
  KXXBeta_ = KXX_.inverse();


  b_need_prepare_ = false;
}

template <typename R>
void GaussianProcessRegression<R>::PrepareRegressionNN(bool force_prepare){
  if(!b_need_prepare_ & !force_prepare)
    return;


  KXX = NNCovFunc(input_data_);
  KXX_ = KXX;
  // add measurement noise
  for(int i=0;i<KXX.cols();i++)
    KXX_(i,i) += sigma_n_*sigma_n_;
  alpha_.resize(output_data_.rows(),output_data_.cols());
  beta_.resize(output_data_.rows(),output_data_.cols()); // New
  // pretty slow decomposition to compute
  //Eigen::FullPivLU<MatrixXr> decomposition(KXX_);
  // this is much much faster:
  Eigen::LDLT<MatrixXr> decomposition(KXX_);
  for (size_t i=0; i < output_data_.rows(); ++i)
    {
      alpha_.row(i) = (decomposition.solve(output_data_.row(i).transpose())).transpose();
      //beta_.row(i) = (decomposition.solve(decomposition.row(i).transpose())).transpose();
    }
  KXXBeta_ = KXX_.inverse();

  

  b_need_prepare_ = false;
}

// This is a slow and and deprecated version that is easier to understand. 
template<typename R>
void GaussianProcessRegression<R>::PrepareRegressionOld(bool force_prepare)
{
  if(!b_need_prepare_ & !force_prepare)
    return;

  KXX = SQEcovFunc(input_data_);
  KXX_ = KXX;
  // add measurement noise
  for(int i=0;i<KXX.cols();i++)
    KXX_(i,i) += sigma_n_*sigma_n_;

  // this is a time theif:
  KXX_ = KXX_.inverse();
  b_need_prepare_ = false;
}

// This is the right way to do it but this code should be refactored and tweaked so that the decompositon is not recomputed unless new training data has arrived. 
template <typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::DoRegression(const VectorXr& inp,bool prepare){
  // if(prepare || b_need_prepare_){
  //   PrepareRegression();
  // }
  // can return immediately if no training data has been added..
  VectorXr outp(output_data_.rows());
  outp.setZero();
  if(n_data_==0)
    return outp;
  
  PrepareRegression(prepare);
  //ok

  outp.setZero();
  KXx = SQEcovFunc(input_data_,inp);
  for (size_t i=0; i < output_data_.rows(); ++i)
    outp(i) = KXx.dot(alpha_.row(i));
    

  return outp;
}

// This is the right way to do it but this code should be refactored and tweaked so that the decompositon is not recomputed unless new training data has arrived. 
template <typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::DoRegressionNN(const VectorXr& inp,bool prepare){
  // if(prepare || b_need_prepare_){
  //   PrepareRegression();
  // }
  // can return immediately if no training data has been added..
  VectorXr outp(output_data_.rows());
  outp.setZero();
  if(n_data_==0)
    return outp;
  
  PrepareRegressionNN(prepare);
  //ok

  outp.setZero();
  KXx = NNCovFunc(input_data_,inp);
  for (size_t i=0; i < output_data_.rows(); ++i)
    outp(i) = KXx.dot(alpha_.row(i));
    

  return outp;
}


// This is the right way to do it but this code should be refactored and tweaked so that the decompositon is not recomputed unless new training data has arrived. This version outputs the variance as well.
template <typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::DoRegressionVariance(const VectorXr& inp,bool prepare){
  // if(prepare || b_need_prepare_){
  //   PrepareRegression();
  // }
  // can return immediately if no training data has been added..
  VectorXr outp(output_data_.rows());

  outp.setZero();
  if(n_data_==0)
    return outp;
  
  PrepareRegression(prepare);
  //ok

  outp.setZero();
  KXx = SQEcovFunc(input_data_,inp);

  // Variance
  VectorXr var(output_data_.rows()); // New

  // For variance calculation.  
  Kxx = SQEcovFunc(inp, inp);
  double sigma_squared = sigma_n_ * sigma_n_;

  

  VectorXr tmp(input_data_.cols());
  // this line is the slow one, hard to speed up further?
  tmp = KXXBeta_ * KXx;


  // Hacking here
  //Eigen::LDLT<MatrixXr> decomposition(KXX_);

  for (size_t i=0; i < output_data_.rows(); ++i){
    outp(i) = KXx.dot(alpha_.row(i));
    //beta_.row(i) = (decomposition.solve(KXx)).transpose(); // Think about if this is right (It is expensive!!! -> think about a way to move it to prepare function..)
    var(i) = tmp.dot(KXx.row(i)); + Kxx(0) + sigma_squared; // Works only for scalars, do

  }


    // Check in the prepare regression section what has to be changed..
  //std::cout << "this stupid output should make the terminal overflow" << std::endl;
  
  var_ = var;
  exists_variance_ = true;  

  return outp;
}

// This is the right way to do it but this code should be refactored and tweaked so that the decompositon is not recomputed unless new training data has arrived. This version outputs the variance as well.
template <typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::DoRegressionNNVariance(const VectorXr& inp,bool prepare){
  // if(prepare || b_need_prepare_){
  //   PrepareRegression();
  // }
  // can return immediately if no training data has been added..
  VectorXr outp(output_data_.rows());

  outp.setZero();
  if(n_data_==0)
    return outp;
  
  PrepareRegressionNN(prepare);
  //ok

  outp.setZero();
  KXx = NNCovFunc(input_data_,inp);

  // Variance
  VectorXr var(output_data_.rows()); // New

  // For variance calculation.  
  Kxx = SQEcovFunc(inp, inp);
  double sigma_squared = sigma_n_ * sigma_n_;

  

  VectorXr tmp(input_data_.cols());
  // this line is the slow one, hard to speed up further?
  tmp = KXXBeta_ * KXx;


  // Hacking here
  //Eigen::LDLT<MatrixXr> decomposition(KXX_);

  for (size_t i=0; i < output_data_.rows(); ++i){
    outp(i) = KXx.dot(alpha_.row(i));
    //beta_.row(i) = (decomposition.solve(KXx)).transpose(); // Think about if this is right (It is expensive!!! -> think about a way to move it to prepare function..)
    var(i) = tmp.dot(KXx.row(i)); + Kxx(0) + sigma_squared; // Works only for scalars, do

  }


    // Check in the prepare regression section what has to be changed..
  //std::cout << "this stupid output should make the terminal overflow" << std::endl;
  
  var_ = var;
  exists_variance_ = true;  

  return outp;
}

template <typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::GetVariance(){
  if (exists_variance_)
  return var_;
  else {
    std::cout << "Warning, Variance not computed, returning zero" << std::endl;
    var_.setZero();
    return var_;
  }
}

template <typename R>
typename GaussianProcessRegression<R>::VectorXr GaussianProcessRegression<R>::DoRegressionOld(const VectorXr& inp,bool prepare){
  if(prepare || b_need_prepare_){
    PrepareRegressionOld();
  }
  VectorXr outp(output_data_.rows());
  outp.setZero();
  KXx = SQEcovFunc(input_data_,inp);
  //KxX = SQEcovFunc(input_data_,inp).transpose();
  VectorXr tmp(input_data_.cols());
  // this line is the slow one, hard to speed up further?
  tmp = KXX_*KXx;
  // the rest is noise in comparison with the above line.
  for(int i=0;i<output_data_.rows();i++){
    outp(i)=tmp.dot(output_data_.row(i));
  }
  return outp;
}

template<typename R>
void GaussianProcessRegression<R>::ClearTrainingData()
{
  input_data_.resize(input_data_.rows(),0);
  output_data_.resize(output_data_.rows(),0);
  b_need_prepare_ = true;
  n_data_ = 0;
}

template<typename R>
typename GaussianProcessRegression<R>::MatrixXr GaussianProcessRegression<R>::SQEcovFunc(MatrixXr x1){
  int nCol = x1.cols();
  MatrixXr retMat(nCol,nCol);
  for(int i=0;i<nCol;i++){
    for(int j=i;j<nCol;j++){
      retMat(i,j)=SQEcovFuncD(x1.col(i),x1.col(j));
      retMat(j,i)=retMat(i,j);
    }
  }
  return retMat;
}

template<typename R>
typename GaussianProcessRegression<R>::MatrixXr GaussianProcessRegression<R>::NNCovFunc(MatrixXr x1){
  int nCol = x1.cols();
  MatrixXr retMat(nCol,nCol);
  for(int i=0;i<nCol;i++){
    for(int j=i;j<nCol;j++){
      // if argument here..
      if (kernel_ == "nn") retMat(i,j)=NNCovFuncD(x1.col(i),x1.col(j));
      if (kernel_ == "c") retMat(i,j)=CCovFuncD(x1.col(i),x1.col(j));
      if (kernel_ == "hy") retMat(i,j)=HYCovFuncD(x1.col(i),x1.col(j));
      if (kernel_ == "ou") retMat(i,j)=OUCovFuncD(x1.col(i),x1.col(j));
      if (kernel_ == "sqe") retMat(i,j)=SQEcovFuncD(x1.col(i),x1.col(j));
      if (kernel_ == "rq") retMat(i,j)=RQCovFuncD(x1.col(i),x1.col(j));
      if (kernel_ == "ousqe") retMat(i,j)=OUSQECovFuncD(x1.col(i),x1.col(j));
      retMat(j,i)=retMat(i,j);
      
    }
  }
  return retMat;
}

template<typename R>
typename GaussianProcessRegression<R>::MatrixXr GaussianProcessRegression<R>::OUCovFunc(MatrixXr x1){
  int nCol = x1.cols();
  MatrixXr retMat(nCol, nCol);
  for(int i=0;i<nCol;i++){
  	for(int j=i;j<nCol;j++){
      retMat(i,j)=OUCovFuncD(x1.col(i),x1.col(j));
      retMat(j,i)=retMat(i,j);
    }
  }
  return retMat;
}

template<typename R>
typename GaussianProcessRegression<R>::MatrixXr GaussianProcessRegression<R>::RQCovFunc(MatrixXr x1){
  int nCol = x1.cols();
  MatrixXr retMat(nCol, nCol);
  for(int i=0;i<nCol;i++){
  	for(int j=i;j<nCol;j++){
      retMat(i,j)=RQCovFuncD(x1.col(i),x1.col(j));
      retMat(j,i)=retMat(i,j);
    }
  }
  return retMat;
}

template<typename R>
void GaussianProcessRegression<R>::Debug()
{
  std::cout<<"input data \n"<<input_data_<<std::endl;
  std::cout<<"output data \n"<<output_data_<<std::endl;
}

#endif /* GAUSSIAN_PROCESS_REGRESSION_HXX */
