//
//  IntrinsicBase.hpp
//  SoftVision
//
//  Created by HouPeihong on 2023/7/21.
//

#ifndef IntrinsicBase_hpp
#define IntrinsicBase_hpp

#include <numeric/numeric.hpp>
#include <camera/cameraCommon.hpp>
#include <camera/IntrinsicInitMode.hpp>
#include <geometry/Pose3.hpp>
#include <string>
//#include <version.hpp>

namespace camera {
/**
 * @brief Basis class for all intrinsic parameters of a camera.
 *
 * Store the image size & define all basis optical modelization of a camera
 */
class IntrinsicBase
{
public:
    explicit IntrinsicBase(unsigned int width, unsigned int height, const std::string& serialNumber = "") :
        _w(width), _h(height), _serialNumber(serialNumber)
    {
    }

    virtual ~IntrinsicBase() = default;
  
    /**
     * @brief Get the lock state of the intrinsic
     * @return true if the intrinsic is locked
     */
    inline bool isLocked() const { return _locked; }
  
    /**
     * @brief Get the intrinsic image width
     * @return The intrinsic image width
     */
    inline unsigned int w() const { return _w; }

    /**
     * @brief Get the intrinsic image height
     * @return The intrinsic image height
     */
    inline unsigned int h() const { return _h; }

    /**
     * @brief Get the intrinsic sensor width
     * @return The intrinsic sensor width
     */
    inline double sensorWidth() const { return _sensorWidth; }

    /**
     * @brief Get the intrinsic sensor height
     * @return The intrinsic sensor height
     */
    inline double sensorHeight() const { return _sensorHeight; }

    /**
     * @brief Get the intrinsic serial number
     * @return The intrinsic serial number
     */
    inline const std::string& serialNumber() const { return _serialNumber; }

    /**
     * @brief Get the intrinsic initialization mode
     * @return The intrinsic initialization mode
     */
    inline EInitMode getInitializationMode() const { return _initializationMode; }

    /**
     * @brief operator ==
     * @param[in] other
     * @return True if equals
     */
    virtual bool operator==(const IntrinsicBase& other) const;

    inline bool operator!=(const IntrinsicBase& other) const { return !(*this == other); }

    /**
     * @brief Projection of a 3D point into the camera plane (Apply pose, disto (if any) and Intrinsics)
     * @param[in] pose The pose
     * @param[in] pt3D The 3d point
     * @param[in] applyDistortion If true apply distrortion if any
     * @return The 2d projection in the camera plane
     */
    virtual Vec2 project(const geometry::Pose3& pose, const Vec4& pt3D, bool applyDistortion = true) const = 0;

    /**
     * @brief Back-projection of a 2D point at a specific depth into a 3D point
     * @param[in] pt2D The 2d point
     * @param[in] applyDistortion If true apply distrortion if any
     * @param[in] pose The camera pose
     * @param[in] depth The depth
     * @return The 3d point
     */
    Vec3 backproject(const Vec2& pt2D, bool applyUndistortion = true, const geometry::Pose3& pose = geometry::Pose3(), double depth = 1.0) const;

    Vec4 getCartesianfromSphericalCoordinates(const Vec3 & pt);

    Eigen::Matrix<double, 4, 3> getDerivativeCartesianfromSphericalCoordinates(const Vec3 & pt);

    /**
     * @brief get derivative of a projection of a 3D point into the camera plane
     * @param[in] pose The pose
     * @param[in] pt3D The 3d point
     * @param[in] applyDistortion If true apply distrortion if any
     * @return The projection jacobian  wrt pose
     */
    virtual Eigen::Matrix<double, 2, 16> getDerivativeProjectWrtPose(const geometry::Pose3& pose, const Vec4& pt3D) const = 0;

    /**
     * @brief get derivative of a projection of a 3D point into the camera plane
     * @param[in] pose The pose
     * @param[in] pt3D The 3d point
     * @param[in] applyDistortion If true apply distrortion if any
     * @return The projection jacobian  wrt point
     */
    virtual Eigen::Matrix<double, 2, 4> getDerivativeProjectWrtPoint(const geometry::Pose3& pose, const Vec4& pt3D) const = 0;

    /**
     * @brief get derivative of a projection of a 3D point into the camera plane
     * @param[in] pose The pose
     * @param[in] pt3D The 3d point
     * @param[in] applyDistortion If true apply distrortion if any
     * @return The projection jacobian wrt params
     */
    virtual Eigen::Matrix<double, 2, Eigen::Dynamic> getDerivativeProjectWrtParams(const geometry::Pose3& pose, const Vec4& pt3D) const = 0;

    /**
     * @brief Compute the residual between the 3D projected point X and an image observation x
     * @param[in] pose The pose
     * @param[in] X The 3D projected point
     * @param[in] x The image observation
     * @return residual
     */
    inline Vec2 residual(const geometry::Pose3& pose, const Vec4& X, const Vec2& x) const
    {
        const Vec2 proj = this->project(pose, X);
        return x - proj;
    }

    /**
     * @brief Compute the residual between the 3D projected point X and an image observation x
     * @param[in] pose The pose
     * @param[in] X The 3D projection
     * @param[in] x The image observation
     * @return residual
     */
    inline Mat2X residuals(const geometry::Pose3& pose, const Mat3X& X, const Mat2X& x) const
    {
        assert(X.cols() == x.cols());
        const std::size_t numPts = x.cols();
        Mat2X residuals = Mat2X::Zero(2, numPts);
        for(std::size_t i = 0; i < numPts; ++i)
        {
            residuals.col(i) = residual(pose, ((const Vec3&)X.col(i)).homogeneous(), x.col(i));
        }
        return residuals;
    }

    /**
     * @brief lock the intrinsic
     */
    inline void lock() { _locked  = true; }

    /**
     * @brief unlock the intrinsic
     */
    inline void unlock() { _locked  = false; }

    /**
     * @brief Set intrinsic image width
     * @param[in] width The image width
     */
    inline void setWidth(unsigned int width) { _w = width; }

    /**
     * @brief Set intrinsic image height
     * @param[in] height The image height
     */
    inline void setHeight(unsigned int height) { _h = height; }

    /**
     * @brief Set intrinsic sensor width
     * @param[in] width The sensor width
     */
    inline void setSensorWidth(double width) { _sensorWidth = width; }

    /**
     * @brief Set intrinsic sensor height
     * @param[in] height The sensor height
     */
    inline void setSensorHeight(double height) { _sensorHeight = height; }
  
    /**
     * @brief Set the serial number
     * @param[in] serialNumber The serial number
     */
    inline void setSerialNumber(const std::string& serialNumber) { _serialNumber = serialNumber; }

    /**
     * @brief Set The intrinsic initialization mode
     * @param[in] initializationMode The intrintrinsic initialization mode enum
     */
    inline void setInitializationMode(EInitMode initializationMode) { _initializationMode = initializationMode; }

    // Virtual members

    /**
     * @brief Polymorphic clone
     */
    virtual IntrinsicBase* clone() const = 0;

    /**
     * @brief Assign object
     * @param[in] other
     */
    virtual void assign(const IntrinsicBase& other) = 0;

    /**
     * @brief Get embed camera type
     * @return EINTRINSIC enum
     */
    virtual EINTRINSIC getType() const = 0;

    /**
     * get a string
     * @return the string describing the intrinsic type
     */
    std::string getTypeStr() const { return EINTRINSIC_enumToString(getType()); }

    /**
     * @brief Get intrinsic parameters
     * @return intrinsic parameters
     */
    virtual std::vector<double> getParams() const = 0;

    /**
     * @brief Get count of intrinsic parameters
     * @return the number of intrinsic parameters
     */
    virtual std::size_t getParamsSize() const = 0;

    /**
     * @brief Update intrinsic parameters
     * @param[in] intrinsic parameters
     * @return true if done
     */
    virtual bool updateFromParams(const std::vector<double>& params) = 0;

//    /**
//     * @brief import intrinsic parameters from external array
//     * @param[in] intrinsic parameters
//     * @param[in] inputVersion input source version (for optional transformation)
//     * @return true if done
//     */
//    virtual bool importFromParams(const std::vector<double>& params, const Version & inputVersion) = 0;

    /**
     * @brief Transform a point from the camera plane to the image plane
     * @param[in] p A point from the camera plane
     * @return Image plane point
     */
    virtual Vec2 cam2ima(const Vec2& p) const = 0;

    /**
     * @brief Transform a point from the image plane to the camera plane
     * @param[in] p A point from the image plane
     * @return Camera plane point
     */
    virtual Vec2 ima2cam(const Vec2& p) const = 0;

    /**
     * @brief Camera model handle a distortion field
     * @return True if the camera model handle a distortion field
     */
    virtual bool hasDistortion() const { return false; }

    /**
     * @brief Add the distortion field to a point (that is in normalized camera frame)
     * @param[in] p The point
     * @return The point with added distortion field
     */
    virtual Vec2 addDistortion(const Vec2& p) const = 0;

    /**
     * @brief Remove the distortion to a camera point (that is in normalized camera frame)
     * @param[in] p The point
     * @return The point with removed distortion field
     */
    virtual Vec2 removeDistortion(const Vec2& p) const = 0;

    /**
     * @brief Return the undistorted pixel (with removed distortion)
     * @param[in] p The point
     * @return The undistorted pixel
     */
    virtual Vec2 get_ud_pixel(const Vec2& p) const = 0;

    /**
     * @brief Return the distorted pixel (with added distortion)
     * @param[in] p The undistorted point
     * @return The distorted pixel
     */
    virtual Vec2 get_d_pixel(const Vec2& p) const = 0;

    /**
     * @brief Set The intrinsic disto initialization mode
     * @param[in] distortionInitializationMode The intrintrinsic distortion initialization mode enum
     */
    virtual void setDistortionInitializationMode(EInitMode distortionInitializationMode) = 0;

    /**
     * @brief Get the intrinsic disto initialization mode
     * @return The intrinsic disto initialization mode
     */
    virtual EInitMode getDistortionInitializationMode() const = 0;

    /**
     * @brief Normalize a given unit pixel error to the camera plane
     * @param[in] value Given unit pixel error
     * @return Normalized unit pixel error to the camera plane
     */
    virtual double imagePlaneToCameraPlaneError(double value) const = 0;

    /**
     * @brief Return true if the intrinsic is valid
     * @return True if the intrinsic is valid
     */
    virtual bool isValid() const { return _w != 0 && _h != 0; }

    /**
     * @brief Return true if this ray should be visible in the image
     * @param ray input ray to check for visibility
     * @return true if this ray is visible theorically
     */
    virtual bool isVisibleRay(const Vec3 & ray) const = 0;

    /**
     * @brief Return true if these pixel coordinates should be visible in the image
     * @param pix input pixel coordinates to check for visibility
     * @return true if visible
     */
    virtual bool isVisible(const Vec2 & pix) const;

    /**
     * @brief Return true if these pixel coordinates should be visible in the image
     * @param pix input pixel coordinates to check for visibility
     * @return true if visible
     */
    virtual bool isVisible(const Vec2f & pix) const;

    /**
     * @brief Assuming the distortion is a function of radius, estimate the
     * maximal undistorted radius for a range of distorted radius.
     * @param min_radius the minimal radius to consider
     * @param max_radius the maximal radius to consider
     * @return the maximal undistorted radius
     */
    virtual float getMaximalDistortion(double min_radius, double max_radius) const;

    /**
     * @brief Generate an unique Hash from the camera parameters (used for grouping)
     * @return Unique Hash from the camera parameters
     */
    virtual std::size_t hashValue() const;

    /**
     * @brief Rescale intrinsics to reflect a rescale of the camera image
     * @param factor a scale factor
     */
    virtual void rescale(float factor);

    /**
     * @brief transform a given point (in pixels) to unit sphere in meters
     * @param pt the input point
     * @return a point on the unit sphere
     */
    virtual Vec3 toUnitSphere(const Vec2 & pt) const = 0;

protected:
    /// initialization mode
    EInitMode _initializationMode = EInitMode::NONE;
    /// intrinsic lock
    bool _locked = false;
    unsigned int _w = 0;
    unsigned int _h = 0;
    double _sensorWidth = 36.0;
    double _sensorHeight = 24.0;
    std::string _serialNumber = nullptr;
};
}

#endif /* IntrinsicBase_hpp */
