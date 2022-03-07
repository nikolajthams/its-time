setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)
options(tikzMetricPackages = c("\\usepackage[utf8]{inputenc}",
                               "\\usepackage[T1]{fontenc}", 
                               "\\usetikzlibrary{calc}",
                               "\\usepackage{amssymb,amsmath}"))

use.tikz <- T
path = "fig-7"
if(use.tikz){tikz(file=paste0(path, ".tex"),width = 4, height = 1.75,
                  packages = c("\\usepackage{tikz}","\\usepackage{amssymb,amsmath}"))}



# Load
df <- read_delim("results.csv", delim=",", col_types=cols(delta="f"))

df.medians <- df %>% group_by(delta, n) %>%
  summarize(error = median(error))


p <- ggplot(df.medians, aes(x = factor(n), y = error, colour=delta, group=delta)) +
    geom_line() + 
    labs(x="$n$", 
         y="$\\operatorname{error}(\\hat\\beta)$",
         colour="Eigenval. $\\Delta$")+
    scale_y_continuous(trans="log10") +
    scale_color_brewer(palette="Dark2") + 
    theme_minimal()

print(p)

if(use.tikz){
  dev.off()
  print(p)
  
  lines <- readLines(con=paste0(path, ".tex"))
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=paste0(path, ".tex"))
  
  ggsave(paste0(path, ".pdf"))
}
