#include "plate_formatter.hpp"
#include <regex>

static const std::map<char, char> CHAR_TO_INT = {
    {'O','0'}, {'I','1'}, {'J','3'}, {'A','4'}, {'G','6'}, {'S','5'}
};
static const std::map<char, char> INT_TO_CHAR = {
    {'0','O'}, {'1','I'}, {'3','J'}, {'4','A'}, {'6','G'}, {'5','S'}
};

std::optional<std::string> format_plate_text(const std::string& text) {
    std::string cleaned;
    for (char c : text) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (std::isalnum(static_cast<unsigned char>(c)) &&
            c != 'X' && c != 'Q' && c != 'W') {
            cleaned += c;
        }
    }

    if (cleaned.size() < 7 || cleaned.size() > 10) return std::nullopt;

    // part1 – first two characters → digits
    std::string part1;
    for (size_t i = 0; i < 2; ++i) {
        char c = cleaned[i];
        auto it = CHAR_TO_INT.find(c);
        part1 += (it != CHAR_TO_INT.end()) ? it->second : c;
    }
    if (!std::all_of(part1.begin(), part1.end(), ::isdigit)) return std::nullopt;

    std::string part2, part3;
    bool found_letters = false;

    for (size_t i = 2; i < cleaned.size(); ++i) {
        char ch = cleaned[i];
        if (std::isalpha(static_cast<unsigned char>(ch))) {
            if (found_letters) return std::nullopt;
            auto it = INT_TO_CHAR.find(ch);
            part2 += (it != INT_TO_CHAR.end()) ? it->second : ch;
        } else if (std::isdigit(static_cast<unsigned char>(ch))) {
            if (!found_letters && !part2.empty()) found_letters = true;
            auto it = CHAR_TO_INT.find(ch);
            part3 += (it != CHAR_TO_INT.end()) ? it->second : ch;
        } else {
            return std::nullopt;
        }
    }

    if (part2.size() >= 1 && part2.size() <= 3 &&
        part3.size() >= 1 && part3.size() <= 4 &&
        std::all_of(part2.begin(), part2.end(), ::isalpha) &&
        std::all_of(part3.begin(), part3.end(), ::isdigit)) {
        return part1 + " " + part2 + " " + part3;
    }
    return std::nullopt;
}