import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class ProgressTracker:
    def __init__(self, data_file="hairline_data.json"):
        self.data_file = data_file
        self.data = self.load_data()
    
    def load_data(self):
        """Load existing data from JSON file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        return {}
                    return json.loads(content)
            except json.JSONDecodeError:
                print("⚠️ Corrupted JSON file detected. Resetting data.")
                return {}
        return {}
    
    def save_data(self):
        """Save data to JSON file safely"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def convert_numpy(self, obj):
        """Recursively convert NumPy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy(i) for i in obj]
        else:
            return obj

    def save_analysis(self, user_id, timestamp, analysis_result):
        """Save analysis results for a user"""
        if user_id not in self.data:
            self.data[user_id] = {}
        
        # ✅ Convert NumPy arrays before saving
        safe_result = self.convert_numpy(analysis_result)
        self.data[user_id][timestamp] = safe_result
        self.save_data()
    
    def generate_report(self, user_id):
        """Generate progress report for a user"""
        if user_id not in self.data or not self.data[user_id]:
            return "No data available for this user."
        
        user_data = self.data[user_id]
        timestamps = sorted(user_data.keys())
        
        if len(timestamps) < 2:
            return "Need at least 2 analyses to track progress."
        
        metrics = {
            'hairline_height': [],
            'forehead_ratio': [],
            'density_score': [],
            'dates': []
        }
        
        for ts in timestamps:
            data = user_data[ts]
            metrics['hairline_height'].append(data['hairline_height'])
            metrics['forehead_ratio'].append(data['forehead_ratio'])
            metrics['density_score'].append(data['density_score'])
            metrics['dates'].append(ts)
        
        progress = self.calculate_progress(metrics)
        self.plot_progress(user_id, metrics)
        return progress
    
    def calculate_progress(self, metrics):
        """Calculate progress metrics"""
        first, last = 0, -1
        hairline_change = metrics['hairline_height'][last] - metrics['hairline_height'][first]
        density_change = metrics['density_score'][last] - metrics['density_score'][first]
        
        report = f"""
        HAIRLINE PROGRESS REPORT
        ========================
        Analysis Period: {metrics['dates'][first]} to {metrics['dates'][last]}
        
        METRICS:
        - Hairline Height Change: {hairline_change:+.3f}
          (Negative = improvement, Positive = recession)
        
        - Density Score Change: {density_change:+.3f}
          (Positive = improvement, Negative = deterioration)
        
        - Overall Progress: {'IMPROVING' if density_change > 0 and hairline_change < 0 else 'STABLE' if abs(hairline_change) < 0.01 else 'NEEDS ATTENTION'}
        
        RECOMMENDATIONS:
        {self.generate_recommendations(hairline_change, density_change)}
        """
        return report
    
    def generate_recommendations(self, hairline_change, density_change):
        """Generate recommendations based on progress"""
        recommendations = []
        if hairline_change > 0.02:
            recommendations.append("- Significant hairline recession detected")
            recommendations.append("- Consider consulting a dermatologist")
        elif hairline_change > 0:
            recommendations.append("- Minor hairline changes observed")
            recommendations.append("- Monitor closely and maintain current routine")
        
        if density_change > 0.1:
            recommendations.append("- Excellent density improvement!")
            recommendations.append("- Continue with current treatment plan")
        elif density_change > 0:
            recommendations.append("- Positive density changes observed")
            recommendations.append("- Treatment appears effective")
        elif density_change < -0.05:
            recommendations.append("- Density decrease detected")
            recommendations.append("- Review treatment approach")
        
        if not recommendations:
            recommendations.append("- No significant changes detected")
            recommendations.append("- Maintain consistent monitoring")
        
        return "\n".join(recommendations)
    
    def plot_progress(self, user_id, metrics):
        """Create progress visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        dates = [d.split('_')[0] for d in metrics['dates']]
        
        ax1.plot(dates, metrics['hairline_height'], 'bo-', linewidth=2)
        ax1.set_title('Hairline Height Over Time')
        ax1.set_ylabel('Normalized Height')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.plot(dates, metrics['forehead_ratio'], 'go-', linewidth=2)
        ax2.set_title('Forehead Ratio Over Time')
        ax2.set_ylabel('Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3.plot(dates, metrics['density_score'], 'ro-', linewidth=2)
        ax3.set_title('Density Score Over Time')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
        
        progress_score = [(1 - h) + d for h, d in zip(metrics['hairline_height'], metrics['density_score'])]
        ax4.plot(dates, progress_score, 'mo-', linewidth=2)
        ax4.set_title('Overall Progress Score')
        ax4.set_ylabel('Progress Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{user_id}_progress_report.png', dpi=300, bbox_inches='tight')
        plt.show()
