#pragma once

//#define CUDA_HOST_DEVICE

/*
 * @struct Range
 * @brief Small CPU and GPU host / device struct descibing a 1d range.
 */
struct Range
{
    unsigned int begin = 0;
    unsigned int end = 0;

    // default constructor
    Range() = default;

    /**
     * @brief Range constructor
     * @param[in] in_begin the range begin index
     * @param[in] in_end the range end index
     */
//    CUDA_HOST_DEVICE Range(unsigned int in_begin,
//                           unsigned int in_end)
//        : begin(in_begin)
//        , end(in_end)
//    {}
//
//    /**
//     * @brief Return true if the given index is contained in the Range.
//     * @param[in] i the given index
//     * @return true if the given index point is contained in the Range
//     */
//    CUDA_HOST_DEVICE inline unsigned int size() const { return end - begin; }
//
//    CUDA_HOST_DEVICE inline bool isEmpty() const { return begin >= end; }
//
//    /**
//     * @brief Return true if the given index is contained in the Range.
//     * @param[in] i the given index
//     * @return true if the given index point is contained in the Range
//     */
//    CUDA_HOST inline bool contains(unsigned int i) const
//    {
//        return ((begin <= i) && (end > i));
//    }
};

inline Range intersect(const Range& a, const Range& b)
{
//    return Range(CUDA_MAX(a.begin, b.begin),
//                 CUDA_MIN(a.end, b.end));
    return Range();//TODO: impl
}


/*
 * @struct ROI
 * @brief Small CPU and GPU host / device struct descibing a rectangular 2d region of interest.
 */
struct ROI
{
    Range x, y;

    // default constructor
    ROI() = default;

    /**
     * @brief ROI constructor
     * @param[in] in_beginX the range X begin index
     * @param[in] in_endX the range X end index
     * @param[in] in_beginY the range Y begin index
     * @param[in] in_endY the range Y end index
     */
//    CUDA_HOST_DEVICE ROI(unsigned int in_beginX,
//                         unsigned int in_endX,
//                         unsigned int in_beginY,
//                         unsigned int in_endY)
//        : x(in_beginX, in_endX)
//        , y(in_beginY, in_endY)
//    {}
//
//    /**
//     * @brief ROI constructor
//     * @param[in] in_rangeX the X index range
//     * @param[in] in_rangeY the Y index range
//     */
//    CUDA_HOST_DEVICE ROI(const Range& in_rangeX,
//                         const Range& in_rangeY)
//        : x(in_rangeX)
//        , y(in_rangeY)
//    {}
//
//    /**
//     * @brief Get the ROI width
//     * @return the X range size
//     */
//    CUDA_HOST_DEVICE inline unsigned int width()  const { return x.size(); }
//
//    /**
//     * @brief Get the ROI height
//     * @return the Y range size
//     */
//    CUDA_HOST_DEVICE inline unsigned int height() const { return y.size(); }
//
//    CUDA_HOST_DEVICE inline bool isEmpty() const { return x.isEmpty() || y.isEmpty(); }
//
//    /**
//     * @brief Return true if the given 2d point is contained in the ROI.
//     * @param[in] in_x the given 2d point X coordinate
//     * @param[in] in_y the given 2d point Y coordinate
//     * @return true if the given 2d point is contained in the ROI
//     */
//    CUDA_HOST inline bool contains(unsigned int in_x, unsigned int in_y) const
//    {
//        return (x.contains(in_x) && y.contains(in_y));
//    }
};

