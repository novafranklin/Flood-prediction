import React, { useState, useEffect } from 'react';
import '@/App.css';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import { CloudRain, Thermometer, Droplets, Gauge, AlertTriangle, CheckCircle, BarChart3, Upload, Brain, Activity } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [dataset, setDataset] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [predictionInput, setPredictionInput] = useState({
    rainfall: '',
    temperature: '',
    humidity: '',
    pressure: '',
    model_type: 'random_forest'
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('dataset');

  const handleGenerateSampleDataset = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/generate-sample-dataset`);
      setDataset(response.data.stats);
      toast.success('Sample dataset generated successfully!');
      setActiveTab('training');
    } catch (error) {
      toast.error('Failed to generate dataset: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload-dataset`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setDataset(response.data.stats);
      toast.success('Dataset uploaded successfully!');
      setActiveTab('training');
    } catch (error) {
      toast.error('Failed to upload dataset: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModel = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/train-model`);
      setMetrics(response.data.metrics);
      toast.success('Models trained successfully!');
      setActiveTab('predict');
    } catch (error) {
      toast.error('Failed to train models: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/predict`, {
        rainfall: parseFloat(predictionInput.rainfall),
        temperature: parseFloat(predictionInput.temperature),
        humidity: parseFloat(predictionInput.humidity),
        pressure: parseFloat(predictionInput.pressure),
        model_type: predictionInput.model_type
      });
      setPredictionResult(response.data.prediction);
      toast.success('Prediction completed!');
    } catch (error) {
      toast.error('Failed to predict: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setPredictionInput(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-blue-100 sticky top-0 z-50">
        <div className="container mx-auto px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-3 rounded-xl shadow-lg">
              <CloudRain className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                Smart Flood Risk Prediction System
              </h1>
              <p className="text-sm text-gray-500">AI-Powered Weather Analysis & Risk Assessment</p>
            </div>
          </div>
        </div>
      </header>

      {/* Alert Banner */}
      {predictionResult && (
        <div className={`${predictionResult.probability > 70 ? 'bg-red-500' : 'bg-green-500'} text-white py-4 px-6 shadow-lg`}>
          <div className="container mx-auto flex items-center justify-center gap-3">
            {predictionResult.probability > 70 ? (
              <AlertTriangle className="w-6 h-6" data-testid="high-risk-icon" />
            ) : (
              <CheckCircle className="w-6 h-6" data-testid="low-risk-icon" />
            )}
            <span className="font-semibold text-lg" data-testid="alert-message">
              {predictionResult.risk_level} - {predictionResult.probability.toFixed(2)}% Probability
            </span>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="container mx-auto px-6 py-12">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full grid-cols-4 bg-white/60 backdrop-blur-sm p-1.5 rounded-xl shadow-sm">
            <TabsTrigger value="dataset" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white rounded-lg transition-all" data-testid="dataset-tab">
              <Upload className="w-4 h-4 mr-2" />
              Dataset
            </TabsTrigger>
            <TabsTrigger value="training" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white rounded-lg transition-all" data-testid="training-tab" disabled={!dataset}>
              <Brain className="w-4 h-4 mr-2" />
              Training
            </TabsTrigger>
            <TabsTrigger value="predict" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white rounded-lg transition-all" data-testid="predict-tab" disabled={!metrics}>
              <Activity className="w-4 h-4 mr-2" />
              Predict
            </TabsTrigger>
            <TabsTrigger value="metrics" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white rounded-lg transition-all" data-testid="metrics-tab" disabled={!metrics}>
              <BarChart3 className="w-4 h-4 mr-2" />
              Metrics
            </TabsTrigger>
          </TabsList>

          {/* Dataset Tab */}
          <TabsContent value="dataset" className="space-y-6">
            <Card className="border-blue-200 shadow-xl bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Upload Dataset</CardTitle>
                <CardDescription>Upload your flood prediction CSV file or generate a sample dataset</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-4">
                  <div className="flex flex-col gap-3">
                    <Label htmlFor="file-upload" className="text-base">Upload CSV File</Label>
                    <Input
                      id="file-upload"
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      className="cursor-pointer"
                      data-testid="file-upload-input"
                    />
                  </div>
                  <div className="flex items-center gap-4">
                    <Separator className="flex-1" />
                    <span className="text-sm text-gray-500">OR</span>
                    <Separator className="flex-1" />
                  </div>
                  <Button
                    onClick={handleGenerateSampleDataset}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-semibold py-6 rounded-xl shadow-lg transition-all hover:shadow-xl"
                    data-testid="generate-sample-btn"
                  >
                    {loading ? 'Generating...' : 'Generate Sample Dataset (1000 records)'}
                  </Button>
                </div>

                {dataset && (
                  <div className="mt-6 space-y-4" data-testid="dataset-preview">
                    <h3 className="text-xl font-semibold">Dataset Preview</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
                        <CardContent className="pt-6">
                          <p className="text-sm text-gray-600">Total Rows</p>
                          <p className="text-3xl font-bold text-blue-600" data-testid="dataset-rows">{dataset.num_rows}</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-gradient-to-br from-indigo-50 to-indigo-100 border-indigo-200">
                        <CardContent className="pt-6">
                          <p className="text-sm text-gray-600">Columns</p>
                          <p className="text-3xl font-bold text-indigo-600" data-testid="dataset-columns">{dataset.num_columns}</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
                        <CardContent className="pt-6">
                          <p className="text-sm text-gray-600">Missing Values</p>
                          <p className="text-3xl font-bold text-purple-600" data-testid="dataset-missing">{Object.values(dataset.missing_values).reduce((a, b) => a + b, 0)}</p>
                        </CardContent>
                      </Card>
                      <Card className="bg-gradient-to-br from-pink-50 to-pink-100 border-pink-200">
                        <CardContent className="pt-6">
                          <p className="text-sm text-gray-600">Features</p>
                          <p className="text-3xl font-bold text-pink-600">4</p>
                        </CardContent>
                      </Card>
                    </div>
                    <div className="bg-white rounded-lg p-4 border overflow-x-auto">
                      <p className="font-semibold mb-2">Sample Data (First 5 rows):</p>
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            {dataset.columns.map(col => (
                              <th key={col} className="px-4 py-2 text-left">{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {dataset.preview.slice(0, 5).map((row, idx) => (
                            <tr key={idx} className="border-b">
                              {dataset.columns.map(col => (
                                <td key={col} className="px-4 py-2">{typeof row[col] === 'number' ? row[col].toFixed(2) : row[col]}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Training Tab */}
          <TabsContent value="training" className="space-y-6">
            <Card className="border-blue-200 shadow-xl bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Train ML Models</CardTitle>
                <CardDescription>Train Random Forest and Logistic Regression models on your dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <Button
                  onClick={handleTrainModel}
                  disabled={loading || !dataset}
                  className="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-semibold py-6 rounded-xl shadow-lg transition-all hover:shadow-xl"
                  data-testid="train-model-btn"
                >
                  {loading ? 'Training Models...' : 'Train Models (Random Forest & Logistic Regression)'}
                </Button>

                {metrics && (
                  <div className="mt-8 space-y-6" data-testid="training-metrics">
                    <h3 className="text-xl font-semibold">Training Results</h3>
                    
                    {/* Random Forest Metrics */}
                    <Card className="bg-gradient-to-br from-green-50 to-emerald-100 border-green-200">
                      <CardHeader>
                        <CardTitle className="text-lg">Random Forest Classifier</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div>
                            <p className="text-sm text-gray-600">Accuracy</p>
                            <p className="text-2xl font-bold text-green-600" data-testid="rf-accuracy">{(metrics.random_forest.accuracy * 100).toFixed(2)}%</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Precision</p>
                            <p className="text-2xl font-bold text-green-600" data-testid="rf-precision">{(metrics.random_forest.precision * 100).toFixed(2)}%</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Recall</p>
                            <p className="text-2xl font-bold text-green-600" data-testid="rf-recall">{(metrics.random_forest.recall * 100).toFixed(2)}%</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">F1-Score</p>
                            <p className="text-2xl font-bold text-green-600" data-testid="rf-f1">{(metrics.random_forest.f1_score * 100).toFixed(2)}%</p>
                          </div>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600 mb-2">ROC AUC Score</p>
                          <Progress value={metrics.random_forest.roc_auc * 100} className="h-3" data-testid="rf-roc-progress" />
                          <p className="text-right text-sm mt-1 font-semibold">{(metrics.random_forest.roc_auc * 100).toFixed(2)}%</p>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Logistic Regression Metrics */}
                    <Card className="bg-gradient-to-br from-blue-50 to-cyan-100 border-blue-200">
                      <CardHeader>
                        <CardTitle className="text-lg">Logistic Regression</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div>
                            <p className="text-sm text-gray-600">Accuracy</p>
                            <p className="text-2xl font-bold text-blue-600" data-testid="lr-accuracy">{(metrics.logistic_regression.accuracy * 100).toFixed(2)}%</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Precision</p>
                            <p className="text-2xl font-bold text-blue-600" data-testid="lr-precision">{(metrics.logistic_regression.precision * 100).toFixed(2)}%</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Recall</p>
                            <p className="text-2xl font-bold text-blue-600" data-testid="lr-recall">{(metrics.logistic_regression.recall * 100).toFixed(2)}%</p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">F1-Score</p>
                            <p className="text-2xl font-bold text-blue-600" data-testid="lr-f1">{(metrics.logistic_regression.f1_score * 100).toFixed(2)}%</p>
                          </div>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600 mb-2">ROC AUC Score</p>
                          <Progress value={metrics.logistic_regression.roc_auc * 100} className="h-3" data-testid="lr-roc-progress" />
                          <p className="text-right text-sm mt-1 font-semibold">{(metrics.logistic_regression.roc_auc * 100).toFixed(2)}%</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Predict Tab */}
          <TabsContent value="predict" className="space-y-6">
            <Card className="border-blue-200 shadow-xl bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Flood Risk Prediction</CardTitle>
                <CardDescription>Enter weather parameters to predict flood risk</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Rainfall */}
                  <div className="space-y-2">
                    <Label htmlFor="rainfall" className="flex items-center gap-2">
                      <CloudRain className="w-4 h-4 text-blue-500" />
                      Rainfall (mm)
                    </Label>
                    <Input
                      id="rainfall"
                      type="number"
                      placeholder="0 - 300"
                      value={predictionInput.rainfall}
                      onChange={(e) => handleInputChange('rainfall', e.target.value)}
                      className="border-blue-200"
                      data-testid="rainfall-input"
                    />
                  </div>

                  {/* Temperature */}
                  <div className="space-y-2">
                    <Label htmlFor="temperature" className="flex items-center gap-2">
                      <Thermometer className="w-4 h-4 text-orange-500" />
                      Temperature (°C)
                    </Label>
                    <Input
                      id="temperature"
                      type="number"
                      placeholder="15 - 40"
                      value={predictionInput.temperature}
                      onChange={(e) => handleInputChange('temperature', e.target.value)}
                      className="border-blue-200"
                      data-testid="temperature-input"
                    />
                  </div>

                  {/* Humidity */}
                  <div className="space-y-2">
                    <Label htmlFor="humidity" className="flex items-center gap-2">
                      <Droplets className="w-4 h-4 text-cyan-500" />
                      Humidity (%)
                    </Label>
                    <Input
                      id="humidity"
                      type="number"
                      placeholder="30 - 100"
                      value={predictionInput.humidity}
                      onChange={(e) => handleInputChange('humidity', e.target.value)}
                      className="border-blue-200"
                      data-testid="humidity-input"
                    />
                  </div>

                  {/* Pressure */}
                  <div className="space-y-2">
                    <Label htmlFor="pressure" className="flex items-center gap-2">
                      <Gauge className="w-4 h-4 text-purple-500" />
                      Atmospheric Pressure (hPa)
                    </Label>
                    <Input
                      id="pressure"
                      type="number"
                      placeholder="980 - 1030"
                      value={predictionInput.pressure}
                      onChange={(e) => handleInputChange('pressure', e.target.value)}
                      className="border-blue-200"
                      data-testid="pressure-input"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Select Model</Label>
                  <div className="flex gap-4">
                    <Button
                      variant={predictionInput.model_type === 'random_forest' ? 'default' : 'outline'}
                      onClick={() => handleInputChange('model_type', 'random_forest')}
                      className={predictionInput.model_type === 'random_forest' ? 'bg-green-500 hover:bg-green-600' : ''}
                      data-testid="rf-model-btn"
                    >
                      Random Forest
                    </Button>
                    <Button
                      variant={predictionInput.model_type === 'logistic_regression' ? 'default' : 'outline'}
                      onClick={() => handleInputChange('model_type', 'logistic_regression')}
                      className={predictionInput.model_type === 'logistic_regression' ? 'bg-blue-500 hover:bg-blue-600' : ''}
                      data-testid="lr-model-btn"
                    >
                      Logistic Regression
                    </Button>
                  </div>
                </div>

                <Button
                  onClick={handlePredict}
                  disabled={loading || !predictionInput.rainfall || !predictionInput.temperature || !predictionInput.humidity || !predictionInput.pressure}
                  className="w-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-semibold py-6 rounded-xl shadow-lg transition-all hover:shadow-xl"
                  data-testid="predict-btn"
                >
                  {loading ? 'Predicting...' : 'Predict Flood Risk'}
                </Button>

                {predictionResult && (
                  <Alert className={`${predictionResult.probability > 70 ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'} mt-6`} data-testid="prediction-result">
                    <AlertDescription className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-lg font-semibold">{predictionResult.risk_level}</span>
                        {predictionResult.probability > 70 ? (
                          <AlertTriangle className="w-6 h-6 text-red-500" />
                        ) : (
                          <CheckCircle className="w-6 h-6 text-green-500" />
                        )}
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 mb-2">Probability</p>
                        <Progress 
                          value={predictionResult.probability} 
                          className={`h-4 ${predictionResult.probability > 70 ? '[&>div]:bg-red-500' : '[&>div]:bg-green-500'}`}
                          data-testid="prediction-probability"
                        />
                        <p className="text-right text-sm mt-1 font-bold" data-testid="probability-value">{predictionResult.probability.toFixed(2)}%</p>
                      </div>
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="space-y-6">
            <Card className="border-blue-200 shadow-xl bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-2xl" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Model Performance Metrics</CardTitle>
                <CardDescription>Detailed analysis of model performance</CardDescription>
              </CardHeader>
              <CardContent>
                {metrics && (
                  <div className="space-y-6">
                    {/* Confusion Matrices */}
                    <div className="grid md:grid-cols-2 gap-6">
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Random Forest - Confusion Matrix</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-2 gap-2 text-center">
                            <div className="p-4 bg-green-100 rounded font-semibold" data-testid="rf-cm-tn">
                              TN: {metrics.random_forest.confusion_matrix[0][0]}
                            </div>
                            <div className="p-4 bg-red-100 rounded font-semibold" data-testid="rf-cm-fp">
                              FP: {metrics.random_forest.confusion_matrix[0][1]}
                            </div>
                            <div className="p-4 bg-red-100 rounded font-semibold" data-testid="rf-cm-fn">
                              FN: {metrics.random_forest.confusion_matrix[1][0]}
                            </div>
                            <div className="p-4 bg-green-100 rounded font-semibold" data-testid="rf-cm-tp">
                              TP: {metrics.random_forest.confusion_matrix[1][1]}
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">Logistic Regression - Confusion Matrix</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-2 gap-2 text-center">
                            <div className="p-4 bg-green-100 rounded font-semibold" data-testid="lr-cm-tn">
                              TN: {metrics.logistic_regression.confusion_matrix[0][0]}
                            </div>
                            <div className="p-4 bg-red-100 rounded font-semibold" data-testid="lr-cm-fp">
                              FP: {metrics.logistic_regression.confusion_matrix[0][1]}
                            </div>
                            <div className="p-4 bg-red-100 rounded font-semibold" data-testid="lr-cm-fn">
                              FN: {metrics.logistic_regression.confusion_matrix[1][0]}
                            </div>
                            <div className="p-4 bg-green-100 rounded font-semibold" data-testid="lr-cm-tp">
                              TP: {metrics.logistic_regression.confusion_matrix[1][1]}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    {/* Model Comparison */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Model Comparison</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div>
                            <div className="flex justify-between mb-2">
                              <span className="text-sm font-medium">Accuracy Comparison</span>
                            </div>
                            <div className="space-y-2">
                              <div>
                                <div className="flex justify-between text-sm mb-1">
                                  <span>Random Forest</span>
                                  <span className="font-semibold">{(metrics.random_forest.accuracy * 100).toFixed(2)}%</span>
                                </div>
                                <Progress value={metrics.random_forest.accuracy * 100} className="h-2 [&>div]:bg-green-500" />
                              </div>
                              <div>
                                <div className="flex justify-between text-sm mb-1">
                                  <span>Logistic Regression</span>
                                  <span className="font-semibold">{(metrics.logistic_regression.accuracy * 100).toFixed(2)}%</span>
                                </div>
                                <Progress value={metrics.logistic_regression.accuracy * 100} className="h-2 [&>div]:bg-blue-500" />
                              </div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="bg-white/80 backdrop-blur-md border-t border-blue-100 mt-16 py-6">
        <div className="container mx-auto px-6 text-center">
          <p className="text-gray-600" style={{ fontFamily: 'Inter, sans-serif' }}>
            Developed with ❤️ | Powered by AI + Machine Learning
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
