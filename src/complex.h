struct Complex
{
	double real, imaginary;

	Complex operator*(const Complex& other)
	{
		return {
			this->real * other.real - this->imaginary * other.imaginary,
			this->real * other.imaginary + this->imaginary * other.real
		};
	}

	Complex square()
	{
		return {
			real * real - imaginary * imaginary,
			2 * real * imaginary
		};
	}

	Complex operator+(const Complex& other)
	{
		return { this->real + other.real, this->imaginary + other.imaginary };
	}
	double norm_sq()
	{
		return this->real * this->real + this->imaginary * this->imaginary;
	}
};