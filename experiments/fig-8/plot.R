setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(tikzDevice)

use.tikz <- F
path = "fig-8"
if(use.tikz){tikz(file=paste0(path, ".tex"),width = 6, height = 2,
                  packages = c("\\usepackage{tikz}","\\usepackage{amssymb,amsmath}"))}

sigma.labels <- c('1' = "do($X_{t} := \\sigma)$", 
                  '5' = "do($X_{t} := 5\\sigma)$",
                  '0' = "do($X_{t} := 0)$"
)

df <- read_delim("results.csv", delim=",", col_types=cols(sigmaout="f", method="f", error="d")) %>% 
  select(-one_of(c("rep_pred", "beta_rel", "error_rel", "...1", "beta"))) %>%
  subset(method != "TB") %>%
  pivot_wider(names_from = "method", values_from = "error") %>%
  select(-one_of(c("rep"))) %>%
  pivot_longer(cols=c("CIV", "NIV"), names_to = "method", values_to="error")

# lims = c(min(c(df$error, df$OLS)), max(c(df$error, df$OLS)))
lims = c(min(c(df$error, df$OLS)), 20)

p <- ggplot(df, aes(x=error, y=OLS, colour = error<OLS, shape=method)) + 
  geom_point() + 
  facet_wrap(~sigmaout, labeller=as_labeller(sigma.labels)) + 
  labs(x="IV MSPE", 
       y="OLS MSPE",
       shape="IV method",
       colour=NULL)+
  geom_abline(intercept=0, slope=1) +
  scale_y_continuous(trans="log10", limits = lims) +
  scale_x_continuous(trans="log10", limits = lims) +
  coord_fixed(ratio = 1, xlim = NULL, ylim = NULL, expand = TRUE, clip = "on") +
  scale_colour_brewer(palette="Dark2", breaks =c("TRUE", "FALSE"), labels=c("IV better", "OLS better")) +
  theme_minimal()

print(p)


if(use.tikz){
  dev.off()
  print(p)
  
  # lines <- readLines(con=paste0(path, ".tex"))
  # lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  # lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  # writeLines(lines,con=paste0(path, ".tex"))
  
  ggsave(paste0(path, ".pdf"))
}
