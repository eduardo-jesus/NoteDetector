#ifndef _LOG_H_
#define _LOG_H_

#include <iostream>
#include <fstream>
#include <string>

class Log {
public:
    Log instance();
    void open(std::string filename);
    void debug(std::string message);
    void close();
private:
    static Log INSTANCE_;
    Log();

    std::ofstream file_;
};

#endif /* _LOG_H_ */
