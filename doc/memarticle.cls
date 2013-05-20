%------------------------------------------------------------
%  Identification
%------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{memarticle}[2012/08/04, Marc Paterno paterno@fnal.gov]

%------------------------------------------------------------
%  Initial code
%------------------------------------------------------------
\RequirePackage{ifthen}
\newboolean{draftmode}
\newboolean{draftwater} % Do a draft watermark?
\newboolean{screen}% Handle pdfscreen ok
\newboolean{notoc}% Do not include a toc
\newboolean{norev}% DO not include a revision number

%------------------------------------------------------------
%  Declaration of options
%------------------------------------------------------------
\DeclareOption{draftmode}{\setboolean{draftmode}{true}}
\DeclareOption{draftwater}{\setboolean{draftwater}{true}}
\DeclareOption{screen}{\setboolean{screen}{true}}
\DeclareOption{notoc}{\setboolean{notoc}{true}}
\DeclareOption{norev}{\setboolean{norev}{true}}
%------------------------------------------------------------
%  Execution of options
%------------------------------------------------------------
\ProcessOptions

%------------------------------------------------------------
%  Package loading
%------------------------------------------------------------
\ifthenelse{\boolean{screen}}
{%
  \LoadClass[12pt, article]{memoir}
}{%
  \LoadClass[11pt, article,letterpaper,twoside]{memoir}
}



\usepackage
  { amsfonts%		extra mathematical fonts
  , bookman%		still the best
%  , palatino%
%  , pxfonts%
%  , mathpazo%
  , booktabs%		"publication quality tables"
  , xcolor%		color capability
  , cpplistings%	display code
  , colornames%
  , graphicx%		incorporate images
  , ifthen%		conditional source
  , xspace%		space character unless punctuation follows
  , varioref%           Smart references
  , microtype%  adjust kerning and more
  }

\usepackage[ragged]{sidecap}

\usepackage%
    [ pdfpagelabels%
    , pdftex%
    , pdfstartview={FitV}%
    , pdfpagemode={}%
    , pdfpagelayout={}%
    ]%
    {hyperref}

\usepackage{memhfixc}%		ensure hyperref/memoir compatability

\ifthenelse{\boolean{screen}}
{
}{%
  \tightlists
}


%------------------------------------------------------------
%  Main code
%------------------------------------------------------------

\ifthenelse{\boolean{draftwater}}
 {%
  \usepackage{draftwatermark} %"DRAFT" watermark
  %\SetWatermarkColor[rgb]{1.0,1.0,0.8}
  \SetWatermarkColor{Yellow}

  \SetWatermarkFontSize{100pt}
  \SetWatermarkText{DRAFT}
 }{}

%\renewcommand{\chapterautorefname}{\S}% use section symbol to refer to chapters
%\renewcommand{\sectionautorefname}{\S}% use section symbol to refer to sections

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% General document parameters and settings:

\ifthenelse{\not \boolean{screen}}
{%
%\settypeblocksize{52\onelineskip}{39pc}{*}  % for 10-pt type
\settypeblocksize{47\onelineskip}{39pc}{*}
\setlrmargins{*}{*}{1}
\setulmargins{*}{*}{1}
}{}


%specify common hyperref options
\hypersetup{%
    anchorcolor=black%
  , bookmarksnumbered=true%
  , bookmarksopen=false%
  , bookmarksopenlevel=1%
  , breaklinks=true%
  , citecolor=blue%
  , colorlinks=true%
  , draft=false%
  , filecolor=blue%
  , linkcolor=blue%
  , linktocpage=true%
  , menucolor=blue%
  , plainpages=false%
  , urlcolor=blue%
  }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Table of contents parameters:

\ifthenelse{\boolean{draftmode}}
{%
  \maxtocdepth{subsubsection}
}{%
  \maxtocdepth{chapter}
}

\maxsecnumdepth{subsubsection}

\setlength{\cftbeforechapterskip}{0in}
\setlength{\cftbeforesectionskip}{0in}
\setlength{\parindent}{0pt}
\setlength  {\parskip     }{.5\baselineskip}
\addtolength{\footnotesep }{.5\baselineskip}
\checkandfixthelayout

\vfuzz2pt	% don't report over-full v-boxes if over-edge is small
\hfuzz2pt	% don't report over-full h-boxes if over-edge is small

\clubpenalty=10000%		discourage widows
\widowpenalty=10000%		discourage widows
\displaywidowpenalty=10000%	discourage widows

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Document's page style
%
% If [draftmode] was specified, we put the document version information on
% the titlepage and on the page headers.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifthenelse{\boolean{norev}}
{%
\newcommand{\docver}{}
\newcommand{\shortdocver}{}
}{%
\newcommand{\docver}{Revision~\docversion}
\newcommand{\shortdocver}{(Rev.~\docversion)}
}

\setlength{\headwidth}{\textwidth}
\makepagestyle   {memarticlepage}
\makerunningwidth{memarticlepage}{\headwidth}
\makeheadrule    {memarticlepage}{\headwidth}{\normalrulethickness}
\makerunningwidth{memarticlepage}{\headwidth}

\ifthenelse{\boolean{draftmode}}
{%
  \makeoddhead     {memarticlepage}%
  {\normalfont\bfseries\brieftitle\ \shortdocver }%
  {}%
  {\normalfont\bfseries\thepage}
  \makeevenhead    {memarticlepage}%
  {\normalfont\bfseries\thepage}%
  {}%
  {\normalfont\bfseries\brieftitle\ \shortdocver}
}{%
  \makeoddhead     {memarticlepage}%
  {\normalfont\bfseries\brieftitle\ \shortdocver}%
  {}%
  {\normalfont\bfseries\thepage}
  \makeevenhead    {memarticlepage}%
  {\normalfont\bfseries\thepage}%
  {}%
  {\normalfont\bfseries\brieftitle\ \shortdocver}
}


\pagestyle{memarticlepage}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% New commands:

\newcommand{\colorrule}[1]%
  {\color{#1}\hrulefill\normalcolor}

\newcommand{\noindentcolorrule}[1]%
  {\noindent\colorrule{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Formatting:

\newcommand{\term}[1]%	format terminology in italicized font
  {\textit{#1}}

\newcommand{\spec}[1]%	format special words in small caps
  {\textsc{\lowercase{#1}}}

\newcommand{\classname}[1]% format class names in italics
  {\textit{#1}}

\newcommand{\conceptname}[1]% format concept names in sans-serif
  {\textsf{#1}}


\newcommand{\important}[1]% important callouts in bold red
  {\textbf{\color{red}#1}}

\newcommand{\code}[1]% inlined (non-C++ code)?, filenames, etc.
  {\texttt{#1}}

\newcommand{\name}[1]% names of products
  {\textbf{#1}}

\newcommand{\cpp}%				'C++'
  {C\kern-0.15ex{+}\kern-0.1ex{+}}

%\newcommand{\ifixme}[1]{{\sffamily\slshape\color{cyan}\textbf{FIXME: } %#1}}
\newcommand{\ifixme}[1]{{\slshape\color{cyan}\textbf{FIXME: } #1}}

% \newenvironment{fixme}%  format notes during document preparation
%   {\par\sffamily\slshape\color{cyan}\textbf{FIXME}}%
%   {} % \rmfamily\upshape\normalcolor

\newenvironment{fixme}%  format notes during document preparation
  {\par\slshape\color{cyan}\textbf{FIXME}}%
  {} % \rmfamily\upshape\normalcolor


\newcommand{\struct}{\cppcode{struct}}
\newcommand{\class}{\cppcode{class}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Symbols:

\newcommand{\middot}{$\cdot$\xspace}% raised dot

%------------------------------------------------------------
% Setup for labels in the description environment; the default
% bold is too strong when using the bookman font.
%------------------------------------------------------------
\renewcommand*{\descriptionlabel}[1]{\hspace\labelsep\normalfont\itshape #1:}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Abbreviations for general use:

\newcommand{\adhoc}{\textit{ad hoc}\xspace}%	ad hoc
\newcommand{\eg   }{\textit{e.g.}\xspace}%	e.g.,
\newcommand{\etal }{\textit{et al.}\xspace}%	et al.
\newcommand{\etc  }{\textit{etc}\xspace}%	etc.
\newcommand{\ie   }{\textit{i.e.}\xspace}%	i.e.,
\newcommand{\qv   }{\textit{q.v.}\xspace}%	q.v.
\newcommand{\sic  }{\textit{sic}\xspace}%	sic
\newcommand{\via  }{\textit{via}\xspace}%	via
\newcommand{\viz  }{\textit{viz.}\xspace}%	viz.

\setlength{\epigraphwidth}{.65\textwidth}
\setlength{\epigraphrule}{0pt}

\epigraphtextposition{flushright}


\newcommand{\topmatter}%
{%
\flushbottom
\raggedbottomsectiontrue
\thispagestyle{empty}
\pagenumbering{arabic}

\begin{flushright}
  { \color[named]{magenta} \HUGE \doctitle  }
  
  { \vskip\baselineskip }
  { \authors } \\
  { \docver }
\end{flushright}

% ----------------------------------------------------------------------
%
% The command \screenflush flushes the current page if, and only if,
% we are in sreen mode.
%
\ifthenelse{\boolean{screen}}%
  {%
    \newcommand{\screenflush}{\clearpage}%
  }{%
    \newcommand{\screenflush}{\relax}%
  }%



% ----------------------------------------------------------------------
%
% Table of Contents:
%

%\ifthenelse{\not \boolean{notoc} \and \not \boolean{screen} }
\ifthenelse{\not \boolean{notoc} }
{%
\noindentcolorrule{Red}
\tableofcontents*
\noindentcolorrule{Red}
}{}
%\nobibintoc

} % end of \topmatter definition

\ifthenelse{ \boolean{screen} }
{%
\setstocksize{6.25in}{8in}
\usepackage[screen]{pdfscreen}
 \margins{0.65in}{0.65in}{0.65in}{0.65in}
 \screensize{6.25in}{8in}
}{}


