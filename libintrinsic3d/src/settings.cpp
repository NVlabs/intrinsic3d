/**
* This file is part of Intrinsic3D.
*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2019, Technical University of Munich. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*    * Redistributions of source code must retain the above copyright
*      notice, this list of conditions and the following disclaimer.
*    * Redistributions in binary form must reproduce the above copyright
*      notice, this list of conditions and the following disclaimer in the
*      documentation and/or other materials provided with the distribution.
*    * Neither the name of NVIDIA CORPORATION nor the names of its
*      contributors may be used to endorse or promote products derived
*      from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <nv/settings.h>

#include <iostream>
#include <sstream>


namespace nv
{

	Settings::Settings()
	{
	}


	Settings::Settings(const std::string &filename)
	{
		if (!filename.empty())
			if (!load(filename))
				std::cerr << "Failed to load settings from file '" << filename << "'!" << std::endl;
	}


	Settings::Settings(const Settings&)
	{
	}


	Settings::~Settings()
	{
	}


    const Settings& Settings::operator=(const Settings& s)
	{
        return s;
	}
	

	template<typename T>
	void Settings::set(const std::string &name, const T &val)
	{
		std::stringstream ss;
		ss << val;
		params_[name] = ss.str();
	}
	template void Settings::set(const std::string &name, const bool &val);
	template void Settings::set(const std::string &name, const char &val);
	template void Settings::set(const std::string &name, const unsigned char &val);
	template void Settings::set(const std::string &name, const short &val);
	template void Settings::set(const std::string &name, const unsigned short &val);
	template void Settings::set(const std::string &name, const int &val);
	template void Settings::set(const std::string &name, const unsigned int &val);
	template void Settings::set(const std::string &name, const long long &val);
	template void Settings::set(const std::string &name, const unsigned long long &val);
	template void Settings::set(const std::string &name, const float &val);
	template void Settings::set(const std::string &name, const double &val);
	template void Settings::set(const std::string &name, const std::string &val);

	template<typename T>
    T Settings::get(const std::string &name) const
	{
		std::string val;
        const auto it = params_.find(name);
        if (it == params_.end())
		{
			std::cerr << "warning: settings parameter '" << name << "' not found!" << std::endl;
			// TODO handle other non-numeric types
			if (std::is_same<T, std::string>::value)
				val = "";
			else
				val = "0";
		}
		else
		{
            val = it->second;
		}

		std::stringstream ss(val);
		T result;
		ss >> result;
		return result;
	}
    template char Settings::get(const std::string &name) const;
    template bool Settings::get(const std::string &name) const;
    template unsigned char Settings::get(const std::string &name) const;
    template short Settings::get(const std::string &name) const;
    template unsigned short Settings::get(const std::string &name) const;
    template int Settings::get(const std::string &name) const;
    template unsigned int Settings::get(const std::string &name) const;
    template long long Settings::get(const std::string &name) const;
    template unsigned long long Settings::get(const std::string &name) const;
    template size_t Settings::get(const std::string &name) const;
    template float Settings::get(const std::string &name) const;
    template double Settings::get(const std::string &name) const;
    template std::string Settings::get(const std::string &name) const;


    bool Settings::empty() const
	{
        return params_.empty();
	}


    bool Settings::exists(const std::string &name) const
	{
        return params_.find(name) != params_.end();
	}


	bool Settings::load(const std::string &filename)
	{
		if (filename.empty())
			return false;
		// open file
		cv::FileStorage fs;
		try
		{
			fs.open(filename, cv::FileStorage::READ);
		}
		catch (...)
		{
			return false;
		}
		if (!fs.isOpened())
			return false;
        // load parameters recursively
        cv::FileNode fs_root = fs.root();
        load(*this, fs_root);
		// close file
		fs.release();
		return true;
	}


    bool Settings::save(const std::string &filename)
	{
		if (filename.empty())
			return false;
		// open file
		cv::FileStorage fs(filename, cv::FileStorage::WRITE);
		if (!fs.isOpened())
			return false;
        // save parameters recursively
		save(*this, fs);
		// close file
		fs.release();
		return true;
	}
	

	void Settings::load(Settings &settings, cv::FileNode &fn)
	{
		// iterate through nodes
		for (cv::FileNodeIterator itr = fn.begin(); itr != fn.end(); ++itr)
		{
			cv::FileNode n = *itr;
			std::string name = n.name();
            if (!n.isMap())
            {
				// load param
                settings.params_[name] = std::string(fn[name]);
			}
		}
	}


    void Settings::save(Settings &settings, cv::FileStorage &fs)
	{
		// save params
		for (auto itr : settings.params_)
			fs << itr.first << itr.second;
	}

} // namespace nv
