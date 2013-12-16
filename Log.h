#ifndef _LOG_H_
#define _LOG_H_

#include <iostream>
#include <fstream>
#include <string>

class Log {
public:
    static Log& instance();
    void open(std::string filename);
    void debug(std::string message);
    void close();
private:
    Log() {};
    Log(Log const&);
    void operator=(Log const&);

    std::ofstream file_;
};

#endif /* _LOG_H_ */
