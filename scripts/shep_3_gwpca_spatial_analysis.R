library(GWmodel)
library(openxlsx)
library(sp)
library(readxl)

# ========== 1. Data preparation ==========
input_file <- "./data/gwpca_calculation_data.xlsx"
output_file <- "./results/GWPCA_analysis_results.xlsx"

if (!file.exists(input_file)) {
  stop("The input file does not exist. Please check the path.: ", input_file)
}

data_df <- read_excel(input_file, col_names = TRUE)

# Set spatial coordinates
if (!inherits(data_df, "Spatial")) {
  coordinates(data_df) <- ~X + Y
}

vars <- c("PLCD", "PLHD", "PLMC", "GVI", "PCRI", "PLLD", "PLLR")

# ========== 2. GWPCA calculation ==========
cat("Selecting bandwidth (Adaptive bandwidth)... \n")
# adaptive = TRUE 
bw_best <- bw.gwpca(data_df, vars = vars, k = 2, robust = FALSE, adaptive = TRUE)

cat("Optimal bandwidth: ", bw_best, "\n")

pca_gw <- gwpca(data_df, vars = vars, bw = bw_best, k = 2, 
                scores = TRUE, robust = FALSE, adaptive = TRUE)

# ========== 3. Results ==========
loadings_df <- as.data.frame(pca_gw$loadings)
scores_df   <- as.data.frame(pca_gw$scores)
local_PV    <- as.data.frame(pca_gw$local.PV)
local_CP    <- as.data.frame(pca_gw$local_CP)

lead_item <- apply(pca_gw$loadings, 1, function(x) {
  vars[which.max(abs(x[1:length(vars)]))]
})

wb <- createWorkbook()

addWorksheet(wb, "Summary")
writeData(wb, "Summary", data.frame(Best_Bandwidth_NN = bw_best))

addWorksheet(wb, "Local_Loadings")
writeData(wb, "Local_Loadings", loadings_df)

addWorksheet(wb, "Principal_Scores")
writeData(wb, "Principal_Scores", scores_df)

addWorksheet(wb, "Local_Variance")
writeData(wb, "Local_Variance", data.frame(
  local_PV, 
  Comp_1_PV = local_PV[,1], 
  Comp_2_PV = local_PV[,2],
  Leading_Item = lead_item
))

addWorksheet(wb, "Local_CP")
writeData(wb, "Local_CP", local_CP)
if (!dir.exists("./results")) dir.create("./results")
saveWorkbook(wb, output_file, overwrite = TRUE)

cat("GWPCA analysis completed ")
