/**
 * \file locale_keeper.h
 * \brief Helper class for set locale and reset locale on dtor
 * \author Miryanov Sergey
 * \date 14.04.2008
 */

#ifndef BS_LOCALE_KEEPER_H_
#define BS_LOCALE_KEEPER_H_

#include <locale.h>
#include <mutex>
#include <string>

namespace blue_sky {

  struct locale_keeper
  {
#ifdef __GLIBC__
    // On glibc ≥2.3, uselocale() is per-thread and thread-safe, unlike
    // setlocale() which affects the entire process globally — no mutex
    // needed: the locale is swapped per-thread in the ctor and restored
    // in the dtor, and a concurrent external setlocale() on another
    // thread cannot affect this thread's uselocale() state.
    locale_t old_locale_;
    locale_t new_locale_;
    int category_;

    locale_keeper (const char *new_name, int category_=LC_ALL)
      : category_ (category_)
    {
      old_locale_ = uselocale(LC_GLOBAL_LOCALE);
      new_locale_ = newlocale(category_, new_name, nullptr);
      if (new_locale_)
        uselocale(new_locale_);
    }

    ~locale_keeper()
    {
      if (new_locale_)
      {
        uselocale(old_locale_);
        freelocale(new_locale_);
      }
    }
#else
    std::string locale_;
    int category_;
    // E-M53: the mutex is held for the keeper's ENTIRE lifetime — from
    // ctor (member-init order: lock_ is acquired before the ctor body
    // swaps the locale) through the caller's parse/write window to the
    // dtor body (which restores the locale while the lock is still held;
    // lock_ is released only after the dtor body runs, during member
    // destruction). The PREVIOUS implementation locked only the swap
    // inside the ctor/dtor and released before the caller's
    // sscanf/fprintf parse window — on macOS/Windows a concurrent
    // external setlocale() could swap the process-global locale in that
    // gap and silently corrupt numeric parse/write (E-M53). The static
    // mutex serializes all locale-dependent parse/write calls on
    // non-glibc platforms; the Python FFI layer already serializes with
    // _hpgl_call_lock, and direct C/C++ callers are now protected too.
    // Note: this makes locale_keeper non-copyable/non-movable (lock_guard
    // member) — all call sites use it as a stack object only.
    std::lock_guard<std::mutex> lock_;

    static std::mutex& get_mutex() {
      static std::mutex m;
      return m;
    }

    locale_keeper (const char *new_name, int category_=LC_ALL)
      : category_ (category_), lock_ (get_mutex())
    {
      // Lock held (lock_ member): save the current locale and swap.
      locale_ = std::string(setlocale(category_, 0));
      setlocale(category_, new_name);
    }

    ~locale_keeper()
    {
      // Lock still held (lock_ is destroyed after the dtor body runs):
      // restore the saved locale.
      setlocale(category_, locale_.c_str());
    }
#endif
  };

}

#endif // #ifndef BS_LOCALE_KEEPER_H_
