#include "Helper.h"

#include <vector>
#include <string>

#include <fstream>

namespace pcv5 {

cv::Matx44f translationMatrix(float x, float y, float z)
{
    cv::Matx44f result = cv::Matx44f::eye();
    result(0, 3) = x;
    result(1, 3) = y;
    result(2, 3) = z;
    return result;
}


cv::Matx44f rotationMatrixX(float radAngle)
{
    cv::Matx44f result = cv::Matx44f::eye();

    result(1, 1) = std::cos(radAngle);
    result(1, 2) = -std::sin(radAngle);
    result(2, 1) = std::sin(radAngle);
    result(2, 2) = std::cos(radAngle);
    return result;
}


cv::Matx44f rotationMatrixY(float radAngle)
{
    cv::Matx44f result = cv::Matx44f::eye();
    result(0, 0) = std::cos(radAngle);
    result(0, 2) = std::sin(radAngle);
    result(2, 0) = -std::sin(radAngle);
    result(2, 2) = std::cos(radAngle);
    return result;
}


cv::Matx44f rotationMatrixZ(float radAngle)
{
    cv::Matx44f result = cv::Matx44f::eye();
    result(0, 0) = std::cos(radAngle);
    result(0, 1) = -std::sin(radAngle);
    result(1, 0) = std::sin(radAngle);
    result(1, 1) = std::cos(radAngle);
    return result;
}





LevenbergMarquardt::LevenbergMarquardt(OptimizationProblem &optimizationProblem, std::unique_ptr<OptimizationProblem::State> initialState) : 
                            m_optimizationProblem(optimizationProblem), m_state(std::move(initialState))
{
    m_newState.reset(m_state->clone());
    m_jacobiMatrix.reset(m_optimizationProblem.createJacobiMatrix());
    
    m_residuals.create(m_optimizationProblem.getNumResiduals(), 1, CV_32F);
    m_newResiduals.create(m_optimizationProblem.getNumResiduals(), 1, CV_32F);
    m_update.create(m_optimizationProblem.getNumUpdateParameters(), 1, CV_32F);
    m_JtR.create(m_optimizationProblem.getNumUpdateParameters(), 1, CV_32F);
    m_diagonal.create(m_optimizationProblem.getNumUpdateParameters(), 1, CV_32F);
    

    m_state->computeResiduals(m_residuals.ptr<float>());
    m_lastError = m_residuals.dot(m_residuals);
//std::cout << "Initial error: " << m_lastError << std::endl;
}

void LevenbergMarquardt::iterate()
{
    m_state->computeJacobiMatrix(m_jacobiMatrix.get());
    m_jacobiMatrix->transposedMultiply(m_JtR.ptr<float>(), m_residuals.ptr<float>());
    
    m_jacobiMatrix->computeDiagJtJ(m_diagonal.ptr<float>());
    m_diagonal *= m_damping;
    
    /*
    auto preconditioner = [](cv::Mat &vec) {
        // todo: Jacobi?
    };
    */
    
    cv::Mat matMulTmp;
    matMulTmp.create(m_optimizationProblem.getNumResiduals(), 1, CV_32F);
    auto matMul = [&](cv::Mat &dst, const cv::Mat &src) {
        dst.create(m_optimizationProblem.getNumUpdateParameters(), 1, CV_32F);
        m_jacobiMatrix->multiply(matMulTmp.ptr<float>(), src.ptr<float>());
        m_jacobiMatrix->transposedMultiply(dst.ptr<float>(), matMulTmp.ptr<float>());
        dst += m_diagonal.mul(src);
    };

    // solve (JtJ + diagonal) . update = JtR for update
    m_update.setTo(0.0f);
    {
        cv::Mat r = m_JtR.clone();
//std::cout << "update " << m_update.t() << std::endl;
//std::cout << "m_JtR " << m_JtR.t() << std::endl;
        cv::Mat p = r.clone();
        float residual = r.dot(r);
        if (residual > 1e-8f) {
            for (unsigned cgIter = 0; cgIter < 100; cgIter++) {
                cv::Mat Ap;
                matMul(Ap, p);
                float pAp = p.dot(Ap);
                float alpha = residual / pAp;
                m_update += alpha * p;
                r -= alpha * Ap;
//std::cout << "pAp " << pAp << std::endl;
//std::cout << "alpha " << alpha << std::endl;
//std::cout << "update " << m_update.t() << std::endl;
                
                float newResidual = r.dot(r);
//std::cout << "CGD res: " << newResidual << std::endl;
                if (newResidual < 1e-8f) 
                    break;
                p *= newResidual / residual;
                p += r;
                residual = newResidual;
            }
        }
#if 0
        {
            cv::Mat y;
            matMul(y, m_update);
            r = y - m_JtR;
            std::cout << "cgd error: " << r.dot(r) << std::endl;
        }
#endif
    }

    m_state->update(m_update.ptr<float>(), m_newState.get());
    
    m_newState->computeResiduals(m_newResiduals.ptr<float>());
    float newError = m_newResiduals.dot(m_newResiduals);

//std::cout << "LM res: " << newError << " damping " << m_damping << std::endl;

    if (newError < m_lastError) {
        m_lastError = newError;
        std::swap(m_newState, m_state);
        std::swap(m_newResiduals, m_residuals);
        m_damping *= 0.9f;
    } else {
        m_damping *= 2.0f;
    }
}



}
