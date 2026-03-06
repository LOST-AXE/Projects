//add in all Stls and exteral header files required
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <format>
#include <fstream>

#include "debug.h"
#include "pgm.h"
#include "dataset.h"
#include "contrastinfo.h"
#include "terminal_graphics.h"

// This function contains our program's core functionality:

void run (std::vector<std::string>& args)
{
//if -v is detected; show all in debug::log () because verbose becomes true in debug.h
  debug::verbose = std::erase (args, "-v");
//error check for size of unix command line arguements - not set to 3 because in instructions it mentioned "which may be acquired in the absence of contrast agent"
  if (args.size() < 2){
    throw std::runtime_error ("missing arguments - expected at least 1 argument");
    }

//inspired by verbose; if a -c is noticed in arguements take the next arguement as contrast file, and query it for ContrastInfo class
    std::string contrast_file; //empty string ready to be filled if -c is present
    bool has_contrast = false; // originally set to false like code in lectures for debug.h
    std::vector<std::string> image_files; 
    //loop through all arguements see if -c is found
    for (unsigned int i = 1; i < args.size(); i++){
        if (args[i]=="-c"){
            if ( i + 1 >= args.size()){
            //if no file was given after -c error the user has given no contrast file
            throw std::runtime_error("no file was given for contrast data after -c");
            }
            //file after -c is contrast file
            contrast_file = args[i + 1];
            // bool is now true
            has_contrast=true;
            // skip next count. i.e dont include -c as one of the image files; it is not
            ++i;
        }
        // if contrast existed its already been saved to contrast file, and -c has been skipped so everything else is image slices, push them back into a vector of strings
        else {
            image_files.push_back(args[i]);
        }
    }
    //check if any images were given
    if (image_files.empty()){
        throw std::runtime_error("no image files were given");
    }
    //use Dataset class methods to load them in 
    Dataset data(image_files);



    


  //pixel interest values hardcoded into the run function; no need for encapsulation as it is really easy to change here and is used as a one off initialization
  int center_x = 74;
  int center_y = 90;
  //want square of 5x5 so 2 in all directions + pixel in centre for all four direction = 5 
  int half_size = 2;

  //create image using Image class, take the width and height of the first slice to do this. made into a class because in future this might be needed multiple times and methods make it much easier to use. 
  Image mask(data.get(0).width(),data.get(0).height());
  //use method for roi in Image class, set everything in ROI to one everything else is zero.
  mask.roi(center_x, center_y, half_size, 1);

//debug to see if mask is being created correctly
int count = 0;

debug::log("Mask with ROI as 1s and everything else as 0s");
//for every y (row) go through each x and add them to the previous one as a string before going to the next row
for (int y = 0; y < mask.height(); ++y){
    std::string row;
    for (int x = 0; x< mask.width(); ++x){
        row += std::to_string(mask.get(x,y));
    }
    //update and display count for easy finding of where roi is meant to be
    count++;
    debug::log(std::format("Row {}: {}", count, row));
}



std::vector<float> lvbp_timecourse;

//find average for every slice, not shown in example in instructions so added as debug
debug::log("Average value in ROI for each frame: ");
//go through each slice; static_cast to avoid errors.
for (int z = 0; z < static_cast<int>(data.size()); ++z){
//uses method in Dataset class to find average of each mask
    float avg = data.average_mask(mask, z);
    // pushback into lvbp_timecourse so it can later be plotted
    lvbp_timecourse.push_back(avg);

    debug::log(std::format("Frame {} has average value of {}", z, avg));
}




//find gradient go through each lvbp_timecourse value and find the difference between next step and current and push it back into gradient
std::vector<float> gradient;
for (size_t y = 0; y < lvbp_timecourse.size() - 1; ++y) {
    gradient.push_back(lvbp_timecourse[y + 1] - lvbp_timecourse[y]);
}







//finding frames; int for dpeakblood because we want frame (discrete value want rounding) and float for speakBlood because we want a exact value.
int dpeakBlood = 0;
float speakBlood = 0.0;
//go through each value, find the biggest one and its index
for (int ii = 0; ii < static_cast<int>(lvbp_timecourse.size()); ++ii) {
    if (lvbp_timecourse[ii] > speakBlood) {
        speakBlood = lvbp_timecourse[ii];
        dpeakBlood = ii;
    }
}
//display everything we've found

std::cerr << "Image at peak contrast concentration:\n";
TG::imshow(TG::magnify(data.get(11),3),0,255);
std::cerr << "Signal timecourse within ROI:\n";
TG::plot().add_line(lvbp_timecourse);
std::cerr << "Gradient of signal timecourse within ROI:\n";
TG::plot().add_line(gradient,3); 
// find first value before peak above 10, use break function to get out once thats done, doesnt continue loop
int darrival = 0;
float Sarrival = 0.0;
float threshold = 10.0;
for (int q = 0; q < dpeakBlood; ++q) {
    if (gradient[q] > threshold) {
        darrival = q;
        Sarrival = lvbp_timecourse[q];
        break;
    }
    
}
//if contrast from beginning comes to play if the bool is true use class ContrastInfo which reads first line as name constrast agent and next as does, encapsulated because it could be used many times with different contrast agents.
if (has_contrast){
        ContrastInfo contrast(contrast_file);
        contrast.print();
    }
    //exact lines in example with values ive found
std::cerr << std::format("Contrast arrival occurs at frame {}, signal intensity: {}\n", darrival, Sarrival);
std::cerr << std::format("Peak contrast concentration occurs at frame {}, signal intensity: {}\n", dpeakBlood, speakBlood);

  
//find G based on formula and display value
float G = (speakBlood - Sarrival) / (dpeakBlood - darrival);
std::cerr << std::format("Temporal gradient of signal during contrast uptake: {}\n", G);

    //part 8 wanted to show everything thats been found below but example pic didnt show this; added to debug to follow instructions and picture
debug::log(std::format("darrival = {}\nSarrival = {}\ndpeakBlood = {}\nSpeakBlood = {}\n", darrival, Sarrival,dpeakBlood, speakBlood));


//original code left over from template from fmri code; i used it to keep track of what i was doing, modified to show centre value for all slices
  int count_debug = 1;
  for (const auto& val : data.get_timecourse (center_x,center_y)){
    debug::log(std::format ("image values at pixel ({},{}) for image {} = {} ", center_x, center_y,count_debug,val));
    count_debug++;
}
}



// skeleton main() function, whose purpose is now to pass the arguments to
// run() in the expected format, and catch and handle any exceptions that may
// be thrown.

int main (int argc, char* argv[])
{
  try {
    std::vector<std::string> args (argv, argv+argc);
    run (args);
  }
  catch (std::exception& excp) {
    std::cerr << "ERROR: " << excp.what() << " - aborting\n";
    return 1;
  }
  catch (...) {
    std::cerr << "ERROR: unknown exception thrown - aborting\n";
    return 1;
  }

  return 0;
}
