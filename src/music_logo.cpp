#include "music_logo.h"
#include <stdio.h>
#include <sys/stat.h>
#include<iostream>

namespace MUSIC_LOGO {

//! This function prints out the program logo
void display_logo(int selector) {
    
    std::cout << "================================================================"  << std::endl;
    std::cout << "|                                                              |"  << std::endl;
    std::cout << "|  ███╗   ███╗██╗   ██╗███████╗██╗ ██████╗   ┌──┬──┬──┐        |"  << std::endl;
    std::cout << "|  ████╗ ████║██║   ██║██╔════╝██║██╔════╝   │▓▓│▓▓│▓▓│        |"  << std::endl;
    std::cout << "|  ██╔████╔██║██║   ██║███████╗██║██║  ███╗  ├──┼──┼──┤  4GPU  |"  << std::endl;
    std::cout << "|  ██║╚██╔╝██║██║   ██║╚════██║██║██║   ██║  │▓▓│▓▓│▓▓│        |"  << std::endl;
    std::cout << "|  ██║ ╚═╝ ██║╚██████╔╝███████║██║╚██████╔╝  ├──┼──┼──┤        |"  << std::endl;
    std::cout << "|  ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝   └──┴──┴──┘        |"  << std::endl;
    std::cout << "|                                                              |"  << std::endl;
    std::cout << "|           GPU-Accelerated MUSIC Hydrodynamics                |"  << std::endl;
    std::cout << "================================================================"  << std::endl;

}

//! This function prints out code desciprtion and copyright information
void display_code_description_and_copyright() {
    std::cout << "MUSIC - a 3+1D viscous relativistic hydrodynamic code for "
              << "heavy ion collisions" << std::endl;
    std::cout << "Copyright (C) 2017  Gabriel Denicol, Charles Gale, Sangyong Jeon, "
              << "Matthew Luzum, Jean-François Paquet, Björn Schenke, Chun Shen"
              << std::endl;
    std::cout << "MUSIG - GPU accelerated MUSIC, Copyright (C) 2026 Joern Putschke"          
              << std::endl;
}

//! This function prints out the welcome message
void welcome_message() {
    display_logo(0);
    display_code_description_and_copyright();
}

}
