/**
 * \file locale_keeper.h
 * \brief Helper class for set locale and reset locale on dtor
 * \author Miryanov Sergey
 * \date 14.04.2008
 */

#ifndef BS_LOCALE_KEEPER_H_
#define BS_LOCALE_KEEPER_H_

#include <locale.h>

namespace blue_sky {

  struct locale_keeper
  {
#ifndef __GLIBC__
    std::string locale_;
    int category_;
#endif
#ifdef __GLIBC__
    locale_t old_locale_;
    locale_t new_locale_;
    int category_;
#endif

    locale_keeper (const char *new_name, int category_=LC_ALL)
      : category_ (category_)
    {
#ifdef __GLIBC__
      // On glibc ≥2.3, uselocale() is per-thread and thread-safe, unlike
      // setlocale() which affects the entire process globally.
      old_locale_ = uselocale(LC_GLOBAL_LOCALE);
      new_locale_ = newlocale(category_, new_name, nullptr);
      if (new_locale_)
        uselocale(new_locale_);
#else
      locale_ = std::string(setlocale(category_, 0));
      setlocale(category_, new_name);
#endif
    }

    ~locale_keeper()
    {
#ifdef __GLIBC__
      if (new_locale_)
      {
        uselocale(old_locale_);
        freelocale(new_locale_);
      }
#else
      setlocale(category_, locale_.c_str());
#endif
    }
  };

}

#endif // #ifndef BS_LOCALE_KEEPER_H_
