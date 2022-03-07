setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)
library(reshape2)
options(tikzMetricPackages = c("\\usepackage[utf8]{inputenc}",
                               "\\usepackage[T1]{fontenc}", 
                               "\\usetikzlibrary{calc}",
                               "\\usepackage{amssymb,amsmath}"))

use.tikz <- T
path = "fig-6-right"
if(use.tikz){tikz(file=paste0(path, ".tex"),width = 3, height = 1.75,
                  packages = c("\\usepackage{tikz}","\\usepackage{amssymb,amsmath}"))}


# Load
df <- read_delim("results.csv", delim=",", col_types=cols(n="f"))

df$logdiff = log10(df$niv_single)-log10(df$niv_all)
maxdiff = max(abs(df$logdiff))

p <- ggplot(df, aes(x=logdiff, fill=factor(niv_all<niv_single))) + 
  geom_histogram(aes(y = stat(count / sum(count))),breaks=seq(-maxdiff,maxdiff,length.out=31), show.legend=T) + 
  labs(x="$\\log_{10} \\operatorname{error}(\\hat{\\beta}_{I^1}) - \\log_{10} \\operatorname{error}(\\hat{\\beta}_{I^{1:3}})$", 
       y=NULL,
       fill=NULL)+
  xlim(c(-1.2,1.2)*maxdiff)+
  # scale_fill_brewer(palette="Dark2") +
  scale_fill_brewer(palette="Dark2", breaks =c("TRUE", "FALSE"), labels=c("$I^{1:3}$ better", "$I^1$ better")) +
  theme_minimal()

print(p)

if(use.tikz){
  dev.off()
  print(p)
  
  ggsave(paste0(path, ".pdf"))
}
