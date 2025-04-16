/*
 * This file is part of [https://github.com/whyb/FastChwHwcConverter].
 * Copyright (C) [2025] [張小凡](https://github.com/whyb)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#pragma once

#include <map>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace whyb {

class DynamicLibraryManager
{
public:
    typedef
#ifdef _WIN32
        HMODULE
#else
        void*
#endif
        LibraryHandle;

public:
    DynamicLibraryManager() {}

    static DynamicLibraryManager* instance()
    {
        static DynamicLibraryManager manager;
        return &manager;
    }

    DynamicLibraryManager(const DynamicLibraryManager&) = delete;
    DynamicLibraryManager& operator=(const DynamicLibraryManager&) = delete;

    ~DynamicLibraryManager()
    {
        for (std::map<std::string, LibraryHandle>::iterator it = libraries_.begin(); it != libraries_.end(); ++it)
        {
            if (it->second)
            {
#ifdef _WIN32
                FreeLibrary(it->second);
#else
                dlclose(it->second);
#endif
            }
        }
    }

    LibraryHandle loadLibrary(std::string libraryName)
    {
        if (libraries_.count(libraryName) == 0)
        {
#ifdef _WIN32
            LibraryHandle handle = LoadLibraryEx(libraryName.c_str(), nullptr,
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS|LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
            LibraryHandle handle = dlopen(libraryName.c_str(), RTLD_LAZY);
#endif
            libraries_[libraryName] = handle;
        }

        return libraries_[libraryName];
    }

    void* getFunction(const std::string& libraryName, const std::string& functionName)
    {
        LibraryHandle libHandle = loadLibrary(libraryName);
        if (!libHandle)
        {
            return nullptr;
        }

#ifdef _WIN32
        return GetProcAddress(static_cast<HMODULE>(libHandle), functionName.c_str());
#else
        return dlsym(libHandle, functionName.c_str());
#endif
    }

    void unloadLibrary(const std::string& libraryName)
    {
        typename std::map<std::string, LibraryHandle>::iterator it = libraries_.find(libraryName);
        if (it != libraries_.end())
        {
#ifdef _WIN32
            FreeLibrary(it->second);
#else
            dlclose(it->second);
#endif
            libraries_.erase(it);
        }
    }

private:
    std::map<std::string, LibraryHandle> libraries_;
};

}
