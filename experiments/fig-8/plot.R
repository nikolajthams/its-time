setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)
options(tikzMetricPackages = c("\\usepackage[utf8]{inputenc}",
                               "\\usepackage[T1]{fontenc}", 
                               "\\usetikzlibrary{calc}",
                               "\\usepackage{amssymb,amsmath}"))

use.tikz <- T
path = "fig-8"

# Load
df <- read_delim("results.csv", delim=",", col_types=cols(strength="f", method="f", n="f"))
plot.df <- df %>% 
  group_by(strength, method, n) %>% 
  summarise(mean.error = mean(error), sd.error=sd(error), median.error=median(error))

p.shared <- ggplot(plot.df, aes(x = n, colour=method, group=method)) +
  labs(x="Sample size", colour='IV method')+
  scale_y_continuous(trans="log10") +
  scale_fill_brewer(palette="Dark2") +
  theme_minimal() 

p.sd <- p.shared + 
  geom_line(mapping=aes(y=sd.error)) + 
  labs(y="$\\operatorname{sd}(\\hat\\beta)$")
p.median <- p.shared + 
  geom_line(mapping=aes(y=mean.error)) + 
  labs(y="$\\operatorname{error}(\\hat\\beta)$")

if(use.tikz){tikz(file=paste0(path, "-left.tex"),width = 3, height = 1.75,
                  packages = c("\\usepackage{tikz}","\\usepackage{amssymb,amsmath}"))}
print(p.median)
if(use.tikz){dev.off()}
if(use.tikz){tikz(file=paste0(path, "-right.tex"),width = 3, height = 1.75,
                  packages = c("\\usepackage{tikz}","\\usepackage{amssymb,amsmath}"))}
print(p.sd)

if(use.tikz){
  dev.off()
  print(p.median)
  print(p.sd)
  }