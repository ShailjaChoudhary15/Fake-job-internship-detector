from ml_predict import predict_ml

registration_fee_keywords = [
    'registration fee', 'joining fee', 'joining charges',
    'advance payment', 'security deposit', 'refundable deposit',
    'training fee', 'training charges', 'kit purchase',
    'laptop security deposit', 'id card processing fee',
    'background verification fee', 'document verification fee',
    'uniform charges', 'agreement charges', 'bond money',
    'pay via upi', 'transfer amount', 'send money',
    'western union', 'wire transfer', 'paytm transfer',
    'gpay transfer', 'phonepe transfer', 'neft transfer',
    'pay to confirm', 'deposit to confirm', 'payment confirmation',
    'fully refundable', 'refund after joining',
    'refund after training', 'money back guarantee',
    'refund after 3 months'
]

government_job_keywords = [
    'sarkari naukri', 'government job guarantee',
    'railway recruitment', 'indian railway job',
    'bank job guarantee', 'ibps guarantee',
    'ssc job guarantee', 'upsc guarantee',
    'defence job', 'army recruitment',
    'police recruitment', 'psu job',
    'government approved', 'ministry approved',
    'central government job', 'state government job',
    'naukri.gov', 'rojgar.gov',
    'sarkarijob.com', 'freejobalert guarantee',
    'appointment letter before interview',
    'offer letter via whatsapp',
    'government job whatsapp',
    'sarkari result guaranteed','government certified programs'
]

work_from_home_keywords = [
    'work from home', 'work from mobile',
    'ghar baithe paise', 'ghar se kaam',
    'earn from home', 'earn from mobile',
    'mobile se paise kamao', 'online earning',
    'like and rate', 'rate and review',
    'product review job', 'amazon review job',
    'flipkart review job', 'app rating job',
    'youtube like job', 'instagram like job',
    'task based earning', 'complete task earn money',
    'daily task work', 'simple task high income',
    'data entry work', 'data entry job',
    'form filling job', 'form filling work',
    'copy paste job', 'copy paste work',
    'typing job from home', 'online typing work',
    'captcha entry job', 'captcha work',
    'earn 1500 per day', 'earn 2000 per day',
    'earn 50000 per month', 'earn 1 lakh per month',
    'earn without investment', 'zero investment',
    'no investment required', 'free registration',
    'earn in spare time', 'part time high income'
]

mlm_keywords = [
    'network marketing', 'mlm', 'multi level marketing',
    'chain business', 'direct selling', 'direct marketing',
    'downline', 'upline', 'referral income',
    'join our team and earn', 'refer and earn',
    'passive income', 'residual income',
    'be your own boss', 'financial freedom',
    'unlimited earning potential', 'no upper limit',
    'amway', 'herbalife', 'modicare distributor',
    'vestige distributor', 'franchise opportunity'
]

fake_company_keywords = [
    'reputed mnc', 'top mnc', 'leading mnc',
    'fortune 500 company', 'international company',
    'foreign company', 'us based company',
    'uk based company', 'dubai company',
    'good company', 'reputed company',
    'well known company', 'established company',
    'tata consultancy hiring', 'infosys recruitment',
    'wipro direct hiring', 'amazon direct hiring',
    'google direct hiring', 'microsoft direct hiring',
    'flipkart direct hiring', 'zomato direct hiring',
    'hr manager calling', 'your resume shortlisted',
    'you have been selected', 'congratulations selected',
    'we found your profile', 'we liked your profile',
    'urgent requirement', 'immediate joining',
    'joining in 2 days', 'join immediately'
]

visa_job_keywords = [
    'abroad job', 'foreign job', 'overseas job',
    'dubai job', 'canada job', 'australia job',
    'uk job', 'usa job', 'gulf job',
    'visa sponsorship guaranteed', 'free visa',
    'visa processing fee', 'visa assistance fee',
    'work permit guaranteed', 'pr guaranteed',
    'gulf recruitment', 'manpower recruitment',
    'overseas placement', 'foreign placement agency',
    'consultation fee visa', 'processing charges abroad'
]
placement_agency_keywords = [
    'placement agency', 'placement consultancy',
    'job consultancy fee', 'placement charges',
    'placement fee', 'cv submission fee',
    'resume submission charges', 'profile registration',
    'interview guarantee fee', 'job guarantee fee',
    '100 percent placement', '100% job guarantee',
    'placement in 7 days', 'job in 15 days',
    'guaranteed interview call', 'guaranteed selection'
]

social_media_scam_keywords = [
    'join telegram group', 'join whatsapp group',
    'telegram channel job', 'whatsapp job offer',
    'contact on whatsapp', 'message on telegram',
    'join group for details', 'link in bio apply',
    'instagram job offer', 'facebook job offer',
    'dm for job details', 'message for details',
    'premium task', 'vip task', 'advance task',
    'task wallet', 'task app earning',
    'crypto task', 'coin task', 'token earning','apply now', 'fill the form']

unrealistic_salary_keywords = [
    'earn 50000 weekly', 'earn 1 lakh weekly',
    '5 lakh per month fresher', 'high salary fresher',
    'salary negotiable no limit', 'unlimited salary',
    'earn crores', 'become crorepati',
    'double your income', 'triple your salary',
    '10x income', 'get rich fast',
    'financial freedom in 6 months',
    'retire in 2 years', 'passive income crores'
]

phishing_keywords = [
    'send aadhar card', 'send pan card',
    'send bank details', 'send account number',
    'share otp', 'otp verification fee',
    'send passport copy', 'send photo id',
    'send personal documents', 'kyc verification fee',
    'bank account verification', 'send cancelled cheque',
    'share ifsc code', 'share account details',
    'noc required', 'noc fee',
]

job_categories = {
    'IT / Computer Science': [
        'software', 'developer', 'programmer', 'coding',
        'python', 'java', 'javascript', 'web development',
        'app development', 'data science', 'machine learning',
        'artificial intelligence', 'cloud', 'devops',
        'cybersecurity', 'database', 'sql', 'networking',
        'it support', 'system admin', 'software engineer',
        'backend', 'frontend', 'fullstack', 'react', 'angular','computer science','automation','debugging'
    ],
    'Medical / Healthcare': [
        'doctor', 'nurse', 'medical', 'pharma', 'hospital',
        'clinical', 'healthcare', 'mbbs', 'bsc nursing',
        'medical representative', 'lab technician',
        'pharmacist', 'dental', 'physiotherapy', 'ayurveda'
    ],
    'Finance / Banking': [
        'accountant', 'finance', 'banking', 'ca', 'chartered',
        'audit', 'tax', 'gst', 'tally', 'mba finance',
        'insurance', 'mutual fund', 'stock market',
        'financial analyst', 'credit analyst', 'loan officer'
    ],
    'Marketing / Sales': [
        'marketing', 'sales', 'business development',
        'digital marketing', 'seo', 'social media',
        'brand manager', 'product marketing', 'field sales',
        'telesales', 'inside sales', 'retail sales'
    ],
    'Engineering': [
        'mechanical', 'civil', 'electrical', 'electronics',
        'chemical', 'production', 'manufacturing',
        'quality control', 'maintenance engineer',
        'autocad', 'design engineer', 'core engineering'
    ],
    'Teaching / Education': [
        'teacher', 'professor', 'lecturer', 'tutor',
        'academic', 'school', 'college', 'university',
        'education', 'training', 'faculty', 'coaching'
    ],
    'Design / Creative': [
        'graphic design', 'ui ux', 'animation',
        'video editing', 'content creation', 'photography',
        'interior design', 'fashion design', 'adobe',
        'photoshop', 'illustrator', 'figma'
    ],
    'HR / Admin': [
        'human resource', 'hr', 'recruitment', 'payroll',
        'admin', 'office management', 'operations',
        'back office', 'data entry admin', 'secretary'
    ]
}

def detect_fake_job(title, description,
                    requirements='', benefits='',
                    company='', user_field=None,
                    salary='', responsibilities=''):

    text = (title + ' ' + description + ' ' +
            requirements + ' ' + benefits + ' ' +
            company + ' ' + salary + ' ' +
            responsibilities).lower()

    results = {
        'risk_level': 'LOW',
        'scam_categories': [],
        'red_flags': [],
        'is_irrelevant': False,
        'irrelevant_reason': '',
        'recommendation': ''
    }

    score = 0

    categories = {
        'Registration/Payment Scam': registration_fee_keywords,
        'Government Job Scam': government_job_keywords,
        'Work From Home Scam': work_from_home_keywords,
        'MLM Scam': mlm_keywords,
        'Fake Company Scam': fake_company_keywords,
        'Visa Scam': visa_job_keywords,
        'Placement Scam': placement_agency_keywords,
        'Social Media Scam': social_media_scam_keywords,
        'Salary Scam': unrealistic_salary_keywords,
        'Phishing Scam': phishing_keywords
    }

    # -------- KEYWORD DETECTION -------- #
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                if category not in results['scam_categories']:
                    results['scam_categories'].append(category)
                if keyword not in results['red_flags']:
                    results['red_flags'].append(keyword)
                score += 1

    # -------- RELEVANCE CHECK -------- #
    if user_field and user_field in job_categories:
        user_keywords = job_categories[user_field]
        job_field_found = None
        user_match = any(kw in text for kw in user_keywords)

        for field, keywords in job_categories.items():
            if field != user_field:
                if any(kw in text for kw in keywords):
                    job_field_found = field
                    break

        if job_field_found and not user_match:
            results['is_irrelevant'] = True
            results['irrelevant_reason'] = (
                f"You are looking for {user_field} jobs but this appears to be a {job_field_found} job"
            )
            score += 2


    ml_pred, ml_prob = predict_ml(text)

    keyword_score = min(score / 10, 1)
    final_score = (0.6 * keyword_score) + (0.4 * ml_prob)

    if final_score < 0.3:
        results['risk_level'] = 'LOW'
        results['recommendation'] =  'SAFE TO APPLY'
    elif final_score < 0.6:
        results['risk_level'] = 'MEDIUM'
        results['recommendation'] = 'BE CAREFUL — Verify before applying'
    elif final_score < 0.8:
        results['risk_level'] = 'HIGH'
        results['recommendation'] = 'LIKELY FAKE — Do not apply'
    else:
        results['risk_level'] = 'VERY HIGH'
        results['recommendation'] = 'DEFINITELY FAKE — Do not apply'

    if results['is_irrelevant']:
        results['risk_level'] = 'IRRELEVANT'
        results['recommendation'] = 'NOT RELEVANT — Do not apply'

    return results

if __name__ == "__main__":

    print("=" * 50)
    print(" FAKE JOB DETECTOR (FULL HYBRID)")
    print("=" * 50)

    title = input("\nEnter job title: ")
    description = input("Enter job description: ")
    requirements = input("Enter requirements: ")
    benefits = input("Enter benefits: ")
    company = input("Enter company: ")
    salary = input("Enter salary: ")
    responsibilities = input("Enter responsibilities: ")

    print("\nSelect your field:")
    fields = list(job_categories.keys())
    for i, field in enumerate(fields, 1):
        print(f"{i}. {field}")

    choice = input("Enter choice: ")
    user_field = fields[int(choice)-1] if choice.isdigit() else None

    result = detect_fake_job(
        title, description, requirements,
        benefits, company, user_field,
        salary, responsibilities
    )

    print("\n" + "=" * 50)
    print(" DETECTION RESULTS")
    print("=" * 50)

    print(f"Final Decision : {result['recommendation']}")
    print(f"Risk Level     : {result['risk_level']}")

    if result['scam_categories']:
        print("\nScam Categories:")
        for c in result['scam_categories']:
            print("-", c)

    if result['red_flags']:
        print(f"\nRed Flags ({len(result['red_flags'])}):")
        for f in result['red_flags'][:10]:
            print("-", f)

    if result['is_irrelevant']:
        print("\nRelevance Issue:")
        print(result['irrelevant_reason'])

    print("=" * 50)