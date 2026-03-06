#include <vector>
#include <string>
#include <format>
#include <stdexcept>

#include "debug.h"
#include "pgm.h"
#include "dataset.h"

void Dataset::load (const std::vector<std::string>& filenames)
{//clear slices
  m_slices.clear();
//check if empty
  if (filenames.empty())
    throw std::runtime_error ("no filenames supplied when loading dataset");
//load slices using pushback and range based for-loop
  for (const auto& fname : filenames)
    m_slices.push_back (load_pgm (fname));

  // check that dimensions all match up, will cause problems if images are of different dimensions
  for (unsigned int n = 1; n < m_slices.size(); ++n) {//index starts at so one so a before always exists to compare to
    if ( (m_slices[n].width() != m_slices[n-1].width()) ||
         (m_slices[n].height() != m_slices[n-1].height()) )
      throw std::runtime_error ("dimensions do not match across slices");
  }

  debug::log (std::format (//-v option see if we've loaded all our images in
      "loaded {} slices of size {}x{}\n",
      m_slices.size(), m_slices[0].width(), m_slices[0].height()));
}





std::vector<int> Dataset::get_timecourse (int x, int y) const
{
  std::vector<int> vals (size());
  for (unsigned int n = 0; n < size(); ++n)
    vals[n] = m_slices[n].get(x,y);
  return vals;
}
//Dataset:: cause i used it here and its methods
float Dataset::average_mask(const Image& mask, int image_slice) const{
//data returns value, from image.h method, private class thats been gotten with getter.

    const Image& image = get(image_slice);
    const std::vector<int>& mask_data = mask.data();
    const std::vector<int>& image_data = image.data();
//see if the mask is the size of the image, mask was initialized based on image and images where checked to be of same size but can never be too safe as this is an MRI and I feel its worth the slight cost in performance 
    if (mask_data.size() != image_data.size()){
        throw std::runtime_error ("mask and image slice dimensions do not match");
} // set float to find exact value and count; method inspired by double sum=0.0; in dna sequencing
        float sum = 0.0;
        int count = 0;
// go through each index, if mask is equal to 1 get the value of that index in image and add it to previous gotten values. update count too so division can happen after to get average
        for (unsigned int i=0; i< mask_data.size(); ++i){
            if (mask_data[i] == 1) {
                sum +=image_data[i];
                ++count;
            }
            
        }
//just a check if count ==0 as in mask was equal to zero or some other shenanigans that didnt cause any of my error checks to kick in. so it must be a valid thing. thus return float 0.0 for average
        if (count == 0){
            return 0.0;
        } 
        
        //had to use static cast because it kept rounding
        return  sum/ static_cast<float>(count);
}


