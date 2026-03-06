#pragma once

#include <array>
#include <vector>
#include <stdexcept>

class Image {
  public:
    Image (int xdim, int ydim) :
      m_dim { xdim, ydim }, m_data (xdim*ydim, 0) { }
//check if sizes match
    Image (int xdim, int ydim, const std::vector<int>& data) :
      m_dim {xdim, ydim }, m_data (data) {
        if (static_cast<int> (m_data.size()) != m_dim[0] * m_dim[1])
          throw std::runtime_error ("dimensions mismatch between image sizes and data vector");
      }

    int width () const { return m_dim[0]; }
    int height () const { return m_dim[1]; }

    const std::vector<int>& data () const { return m_data; }
//row major thing added in class
    int get (int x, int y) const {return m_data[x + m_dim[0]*y];}

    void set (int x, int y, int value) { m_data[x + m_dim[0]*y]= value;}
// roi code, first place to enter hardcodded values in MRI.cpp
    void roi(int center_x, int center_y, int half, int set_value){
    // make sure hard codded values are not out of bounds
     if (center_x + half >= width() || center_x - half < 0 || center_y + half >= height() || center_y - half <0 ){
            throw std::runtime_error("Mask region is outside of image bound");
} //go through each x for each y for each point in roi and set it to hardcodded value, setter
        for (int y = center_y - half; y <= center_y + half; y++){
            for (int x = center_x - half; x <= center_x + half; x++){
            set (x, y, set_value);
               
            }
        }
    }
// added through pure trial and error because TG::show would not work, until operator overloading was used
    int operator()(int x, int y) const {
        return get(x, y);
    }

    
    
  private:
    std::array<int,2> m_dim;
    std::vector<int> m_data;
};
