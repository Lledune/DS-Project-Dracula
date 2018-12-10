rm(list=ls())

setwd("D:/Google Drive/DATS2MS/LINMA2472 Algorithms in data science/Project/Dracula")
data <- read.csv("freqitemsets-clean.csv", sep = ";", header = F)

x <- data[data$V1 == 23,]
x <- x[order(x$V3, decreasing = T),]
png("Images/fisplot.png", 1820, 1080, res = 200)
bp <- barplot(x$V3, ylab = "Character sets", xlab = "Frequency in sentences",
              border = NA, col = "#0049be", horiz = T, main = "Frequency of character sets in chapter XXIII")
inside = (x$V3 <= 40)
text(x = x$V3[inside], y = bp[inside], labels = x$V2[inside], pos = 4, col = "#0049be")
text(x = x$V3[!inside], y = bp[!inside], labels = x$V2[!inside], pos = 2, col = "white")
dev.off()