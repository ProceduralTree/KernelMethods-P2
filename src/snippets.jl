# #+title: Kernel Collocation Excercise
# #+author: Jonathan Ulmer (3545737)
# #+bibliography: ~/org/roam/papers/bibliography.bib
# #+latex_compiler: xelatex
# #+latex_header: \newcommand{\RR}{\mathbb{R}}
# #+latex_header: \newtheorem{remark}{Remark}
# #+latex_header:\usepackage[T1]{fontenc}
# #+latex_header: \usepackage{unicode-math}
# #+latex_header: \setmonofont{DejaVu Sans Mono}[Scale=0.8]
# #+Property: header-args:julia :eval never-export :async t :session *julia* :exports both :tangle src/snippets.jl :comments org


using Plots
f(x) = exp(x)
display(plot(f))
savefig("images/plot.jl");

using Org

@doc org"""
#+begin_src julia :results output
test()
#+end_src

#+RESULTS:
: hello world
"""
function test()
    print("hello world")
    end
