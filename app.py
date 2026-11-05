import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Eğitilmiş modeli yüklemeyi deneyelim (varsa 'best_model.pkl' okunur)
MODEL_PATH = "best_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Eğitilmiş model ('best_model.pkl') başarıyla yüklendi.")
    except Exception as e:
        print(f"⚠️ Model yüklenirken hata oluştu: {e}")
else:
    print("ℹ️ 'best_model.pkl' bulunamadı. Simülasyon modu kullanılacak.")


def estimate_fuel_consumption(data):
    """
    Eğer model .pkl dosyası yoksa veya yüklenemediyse, 
    veri setinizin regresyon parametrelerine dayalı simülasyon hesabı yapar.
    """
    engine_size = float(data.get('engine_size', 2.0))
    gears = int(data.get('gears', 6))
    vehicle_class = int(data.get('vehicle_class', 2))
    fuel_type = data.get('fuel_type', 'X')  # X: Benzin, Z: Premium, E: E85, D: Dizel
    trans_type = data.get('trans_type', 'AS')

    # Baz Tüketim Hesaplama Matrisi
    base_fc = 4.5 + (engine_size * 2.1) - (gears * 0.15) + (vehicle_class * 0.4)
    
    # Yakıt Tipi Düzeltmesi
    fuel_multipliers = {'X': 1.0, 'Z': 1.05, 'E': 1.35, 'D': 0.85, 'N': 1.1}
    base_fc *= fuel_multipliers.get(fuel_type, 1.0)
    
    # Şanzıman Tipi Düzeltmesi
    trans_multipliers = {'AV': 0.93, 'AM': 0.97, 'AS': 1.02, 'M': 0.98, 'A': 1.05}
    base_fc *= trans_multipliers.get(trans_type, 1.0)
    
    return max(4.0, round(base_fc, 2))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.json
        
        engine_size = float(req_data.get('engine_size', 2.0))
        gears = int(req_data.get('gears', 6))
        vehicle_class = int(req_data.get('vehicle_class', 2))
        fuel_type = req_data.get('fuel_type', 'X')
        trans_type = req_data.get('trans_type', 'AS')
        fuel_price = float(req_data.get('fuel_price', 42.50))
        distance = float(req_data.get('distance', 100.0))

        # 1. Model Tahmini veya Simülasyon
        if model is not None:
            # Modelin beklediği DataFrame formatını oluşturuyoruz
            input_df = pd.DataFrame([{
                'Engine_Size': engine_size,
                'Gears': gears,
                'Vehicle_Class_Ord': vehicle_class,
                'Fuel_Type_E': 1 if fuel_type == 'E' else 0,
                'Fuel_Type_N': 1 if fuel_type == 'N' else 0,
                'Fuel_Type_X': 1 if fuel_type == 'X' else 0,
                'Trans_Type_AM': 1 if trans_type == 'AM' else 0,
                'Trans_Type_AS': 1 if trans_type == 'AS' else 0,
                'Trans_Type_AV': 1 if trans_type == 'AV' else 0,
                'Make_Freq': 300.0
            }])
            
            # Sütun hizalaması
            if hasattr(model, 'feature_names_in_'):
                input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
                
            pred_100km = float(model.predict(input_df)[0])
        else:
            pred_100km = estimate_fuel_consumption(req_data)

        # 2. Detaylı Hesaplamalar
        lt_per_km = pred_100km / 100.0
        ml_per_km = lt_per_km * 1000.0
        cost_per_km = lt_per_km * fuel_price
        total_cost = (distance / 100.0) * pred_100km * fuel_price
        total_fuel_lt = (distance / 100.0) * pred_100km

        return jsonify({
            'success': True,
            'pred_100km': round(pred_100km, 2),
            'lt_per_km': round(lt_per_km, 4),
            'ml_per_km': round(ml_per_km, 1),
            'cost_per_km': round(cost_per_km, 2),
            'total_cost': round(total_cost, 2),
            'total_fuel_lt': round(total_fuel_lt, 2),
            'fuel_price': fuel_price,
            'distance': distance
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)