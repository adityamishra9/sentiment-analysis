// sentiment-mobile/metro.config.js
const { getDefaultConfig } = require("expo/metro-config");
module.exports = (() => {
  const cfg = getDefaultConfig(__dirname);
  cfg.resolver.assetExts.push("onnx");
  return cfg;
})();
