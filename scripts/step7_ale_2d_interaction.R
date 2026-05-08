library(readxl)
library(randomForest)
library(ALEPlot)
library(fields) 

# ===================== 1. Environment Configuration =====================
input_file <- "./data/final_analysis_data.xlsx"
output_dir <- "./results/ALE_2D_Interactions"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ===================== 2. Data retrieval =====================
dat <- read_excel(input_file)
dat <- as.data.frame(dat)

vars  <- c("Wall", "Building", "Sky", "Road", "Car", "Fence", "Signboard", "Person", "Bicycle")
X <- dat[, vars]
y <- dat$Scores

pairs <- combn(seq_along(vars), 2)

# ===================== 3. Fitting a Random Forest =====================
set.seed(123)
cat("Training Random Forest for ALE 2D... \n")
rf_model <- randomForest(
  x = X,
  y = y,
  ntree = 500,
  importance = TRUE
)

pred_fun <- function(X.model, newdata) {
  as.numeric(predict(X.model, newdata))
}

# ===================== 4. Generate interactive heatmaps iteratively =====================
for (k in 1:ncol(pairs)) {
  
  j1 <- pairs[1, k]
  j2 <- pairs[2, k]
  var1 <- vars[j1]
  var2 <- vars[j2]
  
  cat(f("Processing Interaction: {var1} x {var2} \n"))

  png(
    filename = file.path(output_dir, paste0("ALE_2D_", var1, "_", var2, ".png")),
    width    = 7,
    height   = 6,
    units    = "in",
    res      = 300
  )

  tryCatch({
    windowsFonts(Times = windowsFont("Times New Roman"))
    par(family = "Times")
  }, error = function(e) {
    par(family = "serif")
  })

  par(mar = c(5, 5, 4, 6), cex.lab = 1.4, cex.axis = 1.2)

  # ALE2D
  ALE_2d <- ALEPlot(
    X        = X,
    X.model  = rf_model,
    pred.fun = pred_fun,
    J        = c(j1, j2),
    K        = 80, # 
    NA.plot  = TRUE
  )

  cols <- hcl.colors(101, "Blue-Red 3")
  zmax <- max(abs(ALE_2d$f.values), na.rm = TRUE)
  zlim <- c(-zmax, zmax)

  fields::image.plot(
    x      = ALE_2d$x.values[[1]],
    y      = ALE_2d$x.values[[2]],
    z      = ALE_2d$f.values,
    col    = cols,
    zlim   = zlim,
    xlab   = var1,
    ylab   = var2,
    main   = paste("Interaction Effect:", var1, "&", var2),
    legend.args = list(text = "ALE Interaction Effect", side = 4, font = 2, line = 3, cex = 1.0)
  )

  contour(
    ALE_2d$x.values[[1]],
    ALE_2d$x.values[[2]],
    ALE_2d$f.values,
    add = TRUE,
    col = adjustcolor("black", alpha.f = 0.3),
    lwd = 0.5
  )

  dev.off()
}

cat(" All 2D ALE plots generated in: ", output_dir, "\n")
