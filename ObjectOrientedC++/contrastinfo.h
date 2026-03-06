#pragma once

#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
#include <format>
//class i made so we have methods to get contrast agent and dose and i wanted to have an excuse to use operator overloading. But again this should be something that should be a class and encapsulated because this can be easily reused. and taken out with -c line.
class ContrastInfo {
//almost everything is public because otherwise its set to private by defualt and there is nothing that I would really want to gaurd here besides the name and dose. although they didnt have an affect on any of the calcultions, i still dont want users to access them and want them to go through my methods to interact with them.
public:
//make class and constructor
    ContrastInfo(const std::string& filename) {
    // see if the file can open
        std::ifstream file(filename);
        if (!file)
            throw std::runtime_error(std::format("cannot open \"{}\"", filename));
    // if it opens, push each line into lines (plural)
        std::vector<std::string> lines;
        std::string line;
        while (file >> line) {
            lines.push_back(line);
        }
    // see if lines is actually exaclty plural, otherwise error 
        if (lines.size() != 2)
            throw std::runtime_error(std::format("\"{}\" has fewer or more than 2 entries", filename));
    //first line is name and second is dose. m_ to indicate method naming
        m_name = lines[0];
        m_dose = std::stof(lines[1]);
    }
    //operator overloading so that name and dose can be later easily derived if the research team wants to code new stuff.
    std::string operator[](int index) const {
        if (index == 0) return m_name;
        if (index == 1) return std::to_string(m_dose);
        //if these two cases dont exist i.e we checked there are two lines but if user types anything out of that range throw an error
        throw std::runtime_error("Index must be either 0 or 1");
            
    
    }
//other ways to get values besides () and print that automatically type line wanted in part 8.
    std::string name() const { return m_name; }
    float dose() const { return m_dose; }

    void print() const {
        std::cout << "Contrast Agent: " << m_name
                  << ", Dose: " << m_dose << " mmol/kg\n";
    }
    private:
        std::string m_name;
        float m_dose;
};
