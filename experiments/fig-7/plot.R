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

# compute lower and upper whiskers
ylim1 = boxplot.stats(log10(df$error))$stats[c(1, 5)]

# p <- ggplot(df, aes(x=delta,  y=error, fill=n)) +
#   geom_boxplot(outlier.alpha=0.1) +
#   labs(x="$\\Delta$", y="$\\hat{\\mathbb{E}}\\|\\hat\\beta - \\beta\\|^2_2$", fill="Sample size")+
#   scale_y_continuous(trans="log10") +
#   coord_cartesian(ylim = (10**(ylim1*1.1)))+
#   theme_minimal()
df.means <- df %>% group_by(delta, n) %>%
  summarize(upper = quantile(error, 0.975),
            lower = quantile(error, 0.025),
            error = median(error))


p <- ggplot(df.means, aes(x = factor(n), y = error, colour=delta, group=delta)) +
    geom_line() + 
    labs(x="$n$", 
         y="$\\operatorname{error}(\\hat\\beta)$",
           #"$\\operatorname{median}\\|\\hat\\beta - \\beta\\|^2_2$", 
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
