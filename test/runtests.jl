using SarmaHybrid
using Test
using NumericalIntegration

@testset "SarmaHybrid.jl" begin
<<<<<<< HEAD
    @test isapprox(SarmaHybrid.v1_f(3.0, 15.0, 500.0, 1.0), 75.75) 
    @test isapprox(SarmaHybrid.v1_f(3.0, 15.0, 500.0, 0.1), 3.0/11.0)
    
    @test isapprox(SarmaHybrid.v2_f(20.0, 100.0, 1.0 ), 1.0)
    @test isapprox(SarmaHybrid.v2_f(20.0, 100.0, 10.0), 200.0/11.0)

    @test isapprox(SarmaHybrid.v3n_f(7.0, 20.0, 40.0, 18.0, 1.0 ), 0.7/12.0)
    @test isapprox(SarmaHybrid.v3n_f(7.0, 20.0, 40.0, 18.0, 10.0), 70.0/651.0)

    @test isapprox(SarmaHybrid.v4n_f(14.5, 40.0, 60.0, 27.0, 1.0 ), 4.35/24.0)
    @test isapprox(SarmaHybrid.v4n_f(14.5, 40.0, 60.0, 27.0, 10.0), 435.0/1581.0)
    
    @test isapprox(SarmaHybrid.v5_f(1.0, 80.0, 400.0, 1.0 ), 0.4/25.0)
    @test isapprox(SarmaHybrid.v5_f(1.0, 80.0, 400.0, 10.0), 40.0/241.0)

    @test isapprox(SarmaHybrid.v6_f(10.0, 20.0, 40.0, 1.0), 0.2/4.0)
    @test isapprox(SarmaHybrid.v6_f(10.0, 20.0, 40.0, 10.0), 20.0/31.0)

    @test isapprox(SarmaHybrid.v7_f(20.0, 40.0, 60.0, 1.0 ), 4.0/6.0)
    @test isapprox(SarmaHybrid.v7_f(20.0, 40.0, 60.0, 10.0), 400.0/51.0)

    @test isapprox(SarmaHybrid.v8_f(30.0, 20.0, 40.0, 1.0 ), 3.0/4.0)
    @test isapprox(SarmaHybrid.v8_f(30.0, 20.0, 40.0, 10.0), 300.0/31.0)

    @test isapprox(SarmaHybrid.v9_f(100.0, 40.0, 60.0, 1.0 ), 4.0/6.0)
    @test isapprox(SarmaHybrid.v9_f(100.0, 40.0, 60.0, 10.0), 400.0/51.0)

    @test isapprox(SarmaHybrid.v10_f(50.0, 40.0, 60.0, 1.0 ), 2.0/6.0)
    @test isapprox(SarmaHybrid.v10_f(50.0, 40.0, 60.0, 10.0), 200.0/51.0)

    @test SarmaHybrid.gaussian(Inf, 0.0, 1.0) == 0.0
    @test SarmaHybrid.gaussian(-Inf, 0.0, 1.0) == 0.0
    @test isapprox(integrate((-10:0.01:10), SarmaHybrid.gaussian.(-10:0.01:10, 0.0, 1.0)), 1.0)
    @test isapprox(integrate((-10:0.01:0), SarmaHybrid.gaussian.(-10:0.01:0, 0.0, 1.0)), 0.5)
    @test isapprox(integrate((0:0.01:10), SarmaHybrid.gaussian.(0:0.01:10, 0.0, 1.0)), 0.5)

    @test SarmaHybrid.model_s2(repeat([0], 12), 1) == repeat([0], 12)

=======
    # Write your tests here.
>>>>>>> e798cee8f275ae677bc26cdd602a5768cba10f6a
end
