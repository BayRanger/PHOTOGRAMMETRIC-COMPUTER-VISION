#ifndef HELPER_H
#define HELPER_H


#include <opencv2/opencv.hpp>


namespace pcv5 {


/// Interpretation of the internal and external parts of a projection matrix.
struct ProjectionMatrixInterpretation
{
    /// Principal distance or focal length
    float principalDistance;
    /// Skew as an angle and in degrees
    float skew;
    /// Aspect ratio of the pixels
    float aspectRatio;
    /// Location of principal point in image (pixel) coordinates
    cv::Vec2f principalPoint;
    /// Camera rotation angle 1/3
    float omega;
    /// Camera rotation angle 2/3
    float phi;
    /// Camera rotation angle 3/3
    float kappa;
    /// 3D camera location in world coordinates
    cv::Vec3f cameraLocation;
};




template<int dim>
cv::Vec<float, dim-1> hom2eucl(const cv::Vec<float, dim> &point) {
    cv::Vec<float, dim-1> result;
    for (unsigned i = 0; i < dim-1; i++)
        result(i) = point(i) / point(dim-1);
    return result;
}

cv::Matx44f translationMatrix(float x, float y, float z);
cv::Matx44f rotationMatrixX(float radAngle);
cv::Matx44f rotationMatrixY(float radAngle);
cv::Matx44f rotationMatrixZ(float radAngle);


class OptimizationProblem
{
    public:
        class JacobiMatrix {
            public:
                virtual ~JacobiMatrix() = default;
                
                virtual void multiply(float * __restrict dst, const float * __restrict src) const = 0;
                virtual void transposedMultiply(float * __restrict dst, const float * __restrict src) const = 0;
                virtual void computeDiagJtJ(float * __restrict dst) const = 0;
        };

        class State {
            public:
                virtual ~State() = default;

                virtual State* clone() const = 0;
                virtual void computeResiduals(float *residuals) const = 0;
                virtual void computeJacobiMatrix(JacobiMatrix *dst) const = 0;
                virtual void update(const float *update, State *dst) const = 0;
        };
        
        inline unsigned getNumUpdateParameters() const { return m_numUpdateParameters; }
        inline unsigned getNumResiduals() const { return m_numResiduals; }
        
        virtual JacobiMatrix* createJacobiMatrix() const = 0;
    protected:
        unsigned m_numUpdateParameters;
        unsigned m_numResiduals;
        std::unique_ptr<State> m_state;
};

class LevenbergMarquardt
{
    public:
        LevenbergMarquardt(OptimizationProblem &optimizationProblem, std::unique_ptr<OptimizationProblem::State> initialState);
        
        void iterate();

        inline float getLastError() const { return m_lastError; }
        inline float getDamping() const { return m_damping; }
        
        const OptimizationProblem::State *getState() const { return m_state.get(); }
    protected:
        OptimizationProblem &m_optimizationProblem;
        
        std::unique_ptr<OptimizationProblem::State> m_state;
        std::unique_ptr<OptimizationProblem::State> m_newState;
        std::unique_ptr<OptimizationProblem::JacobiMatrix> m_jacobiMatrix;
        cv::Mat m_diagonal;
               
        cv::Mat m_residuals;
        cv::Mat m_newResiduals;
        cv::Mat m_JtR;
        cv::Mat m_update;
        
        float m_lastError = 0.0f;
        float m_damping = 1.0f;
        
        cv::Mat m_conditioner;
};


}


#endif // HELPER_H
