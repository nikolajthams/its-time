setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)
options(tikzMetricPackages = c("\\usepackage[utf8]{inputenc}",
                               "\\usepackage[T1]{fontenc}", 
                               "\\usetikzlibrary{calc}",
                               "\\usepackage{amssymb,amsmath}"))

use.tikz <- T
path = "fig-8"
if(use.tikz){tikz(file=paste0(path, ".tex"),width = 4, height = 1.75,
                  packages = c("\\usepackage{tikz}","\\usepackage{amssymb,amsmath}"))}

# Load
df <- read_delim("results.csv", delim=",", col_types=cols(strength="f", method="f"))

# Geom jitter below is downsampled for better tikz-plotting
p <- ggplot(df, aes(x = method, y = error, fill=n)) +
  geom_violin(outlier.alpha=0.3) + 
  geom_jitter(data = df[sample(nrow(df), 1200), ], position=position_jitterdodge(jitter.width=0.5, dodge.width=0.9), alpha=0.05, size=.05) + 
  labs(x="Estimator", 
       y="$\\operatorname{error}(\\hat\\beta)$",
       fill="Sample size")+
  scale_y_continuous(trans="log10") +
  scale_fill_brewer(palette="Dark2") +
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