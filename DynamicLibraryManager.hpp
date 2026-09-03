/*
 * This file is part of [https://github.com/whyb/FastChwHwcConverter].
 * Copyright (C) [2025-2026] [張小凡](https://github.com/whyb)
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
#include <mutex>
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
        std::lock_guard<std::mutex> lock(librariesMutex_);
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

    // Returns the most recent load/symbol error.  The library does not write
    // diagnostics directly to std::cerr, so callers can present them as needed.
    std::string getLastError() const
    {
        std::lock_guard<std::mutex> lock(errorMutex_);
        return lastError_;
    }

    void clearError()
    {
        std::lock_guard<std::mutex> lock(errorMutex_);
        lastError_.clear();
    }

    LibraryHandle loadLibrary(const std::string& libraryName)
    {
        {
            std::lock_guard<std::mutex> lock(librariesMutex_);
            auto existing = libraries_.find(libraryName);
            if (existing != libraries_.end())
            {
                return existing->second;
            }
        }

#ifdef _WIN32
        LibraryHandle handle = LoadLibraryEx(libraryName.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS|LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!handle) {
            setLastError("loadLibrary failed for '" + libraryName + "', system code: " +
                         std::to_string(GetLastError()) + ".");
            return nullptr;
        }
#else
        LibraryHandle handle = dlopen(libraryName.c_str(), RTLD_LAZY);
        if (!handle) {
            const char* reason = dlerror();
            setLastError(std::string("loadLibrary failed for '") + libraryName + "'" +
                         (reason ? std::string(": ") + reason : std::string(".")));
            return nullptr;
        }
#endif
        {
            std::lock_guard<std::mutex> lock(librariesMutex_);
            auto existing = libraries_.find(libraryName);
            if (existing != libraries_.end())
            {
                if (handle) {
#ifdef _WIN32
                    FreeLibrary(handle);
#else
                    dlclose(handle);
#endif
                }
                return existing->second;
            }
            libraries_[libraryName] = handle;
        }
        return handle;
    }

    void* getFunction(const std::string& libraryName, const std::string& functionName)
    {
        LibraryHandle libHandle = loadLibrary(libraryName);
        if (!libHandle)
        {
            return nullptr;
        }

#ifdef _WIN32
        void* function = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(libHandle), functionName.c_str()));
#else
        void* function = dlsym(libHandle, functionName.c_str());
#endif
        if (!function)
        {
            setLastError("Failed to resolve symbol '" + functionName + "' in '" + libraryName + "'.");
        }
        return function;
    }

    void unloadLibrary(const std::string& libraryName)
    {
        LibraryHandle handle = nullptr;
        {
            std::lock_guard<std::mutex> lock(librariesMutex_);
            auto it = libraries_.find(libraryName);
            if (it == libraries_.end())
            {
                return;
            }
            handle = it->second;
            libraries_.erase(it);
        }

        if (handle)
        {
#ifdef _WIN32
            FreeLibrary(handle);
#else
            dlclose(handle);
#endif
        }
    }

private:
    void setLastError(const std::string& message) const
    {
        std::lock_guard<std::mutex> lock(errorMutex_);
        lastError_ = message;
    }

    std::map<std::string, LibraryHandle> libraries_;
    mutable std::mutex librariesMutex_;
    mutable std::mutex errorMutex_;
    mutable std::string lastError_;
};

}
