/**
 * \file bs_assert.h
 * \brief smart assertation, based on Alexandrescu's ideas
 * \author Sergey Miryanov
 * \date 14.05.2008
 */
#ifndef BS_ASSERT_H_
#define BS_ASSERT_H_

#if defined (_WIN32)
    #if defined(_M_X64) || defined(__x86_64__)
        // x64: inline asm not supported, use intrinsic debugbreak
        #define BREAK_HERE __debugbreak()
    #else
        // x86: inline asm is supported
        #define BREAK_HERE __asm { int 3 }
    #endif
#elif defined(__aarch64__) || defined(__arm64__)
    // ARM64 / Apple Silicon: use brk instruction
    #define BREAK_HERE __asm__ volatile("brk #0")
#elif defined(__arm__)
    // ARM32: use bkpt instruction or builtin trap
    #define BREAK_HERE __builtin_trap()
#elif defined(__x86_64__) || defined(__i386__) || defined(__i686__)
    // x86/x86_64 Linux/macOS: use int3
    #define BREAK_HERE __asm__ __volatile__ ("int $0x3")
#else
    // Unknown architecture: fallback to builtin trap
    #define BREAK_HERE __builtin_trap()
#endif

#include <sstream>

namespace blue_sky {

  namespace bs_assert {

    //! forward declaration
    class assert_factory;

    /** 
     * \brief perform assert actions
     * */
    class BS_API_PLUGIN asserter 
    {

    public:

      /** 
       * \brief user reaction on assert
       * */
      enum assert_state 
      {
        STATE_KILL,                               //! terminate process
        STATE_BREAK,                              //! break into file where assertion failed
        STATE_IGNORE,                             //! ignore current assert
        STATE_IGNORE_ALL,                         //! ignore current and all following asserts
      };

    public:

      asserter (const char *file, int line, const char *cond_str)
        : line (line)
	, file (file)
	, cond (true)
        , cond_s (cond_str)
        , var_list ("")
      {
        ASSERTER_A = this;
        ASSERTER_B = this;
      }

      asserter (bool cond, const char *file, int line, const char *cond_str)
        : line (line) 
	, file (file)
	, cond (cond)        
	, cond_s (cond_str)
        , var_list ("")
      {
        ASSERTER_A = this;
        ASSERTER_B = this;
      }

      virtual ~asserter ()
      {
      }

      virtual assert_state ask_user () const;

      virtual bool handle () const;

      static asserter 
      workaround (const char * file_, int line_, const char *cond_str)
      {
        return asserter (file_, line_, cond_str);
      }

      asserter *make (bool cond); 
      
      template <class T> asserter *
      add_var (const T &t, const std::string &name)
      {
        std::ostringstream oss;
        oss << name << " = " << t << "\n";
        var_list = var_list + oss.str();

        return this;
      }

      inline bool &
      ignore_all () const
      {
        static bool ignore_all_ = false;
        return ignore_all_;
      }

      static void 
      set_factory (assert_factory *f)
      {
        factory() = f;
      }

      virtual const char *what ()const noexcept
      {
        return "asserter";
      }

      static assert_factory *&
      factory ()
      {
        static assert_factory *factory_ = 0;
        return factory_;
      }

    public:

      asserter                *ASSERTER_A;
      asserter                *ASSERTER_B;

    public:
      int                     line;
      const char              *file;
      bool                    cond;
			const char              *cond_s;
      std::string             var_list;
      //static assert_factory   *factory;
    };  

    class BS_API_PLUGIN assert_factory 
    {
    public:

      virtual ~assert_factory () {}

      virtual asserter *make (bool b, const char *file, int line, const char *cond_str);
    };

    struct BS_API_PLUGIN assert_wrapper
    {
      assert_wrapper (asserter *pa)
      {
        if (pa && !pa->handle ())
          {
            BREAK_HERE;
          }

        delete pa;
      }
    };

  } // namespace bs_assert

#ifdef _DEBUG
#define BS_ASSERT(cond)                                         \
  if (false) ; else												\
	blue_sky::bs_assert::assert_wrapper wrapper_ = blue_sky::bs_assert::asserter::workaround(__FILE__, __LINE__, (#cond)).make(!!(cond))->ASSERTER_A
#else
#define BS_ASSERT(cond)                                         \
  if (true) ; else												\
	blue_sky::bs_assert::assert_wrapper wrapper_ = blue_sky::bs_assert::asserter::workaround(__FILE__, __LINE__, (#cond)).make(!!(cond))->ASSERTER_A
#endif

#define ASSERTER_A(x)           ASSERTER_OP_(x, B)
#define ASSERTER_B(x)           ASSERTER_OP_(x, A)
#define ASSERTER_OP_(x, next)   ASSERTER_A->add_var ((x), #x)->ASSERTER_##next

} // namespace blue_sky


#endif  // #ifndef BS_ASSERT_H_

