/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

// Uncomment if you experience unreasonably long build times (only compile kernels for the default queue sizes)
// Additionally, you can specify the exact CUDA_ARCHITECTURE in the CMakeLists.txt (default "70;75;86")
#define STOPTHEPOP_FASTBUILD

#include <vector>
#include <functional>

#include "json/json.hpp"

namespace CudaRasterizer
{
	enum DebugVisualization
	{
		Disabled = 0,
		GaussianCountPerTile = 1,
		GaussianCountPerPixel = 2,
		Transmittance = 3,
		Depth = 4,
		Opacity = 5,
		Distortion = 6,
		EXTENT_LOSS = 7,
		Depth_Index = 8,
		DEPTH_EXTENT = 9,
	};
	
	
	inline std::string toString(DebugVisualization m) {
		switch(m)
		{
			case DebugVisualization::GaussianCountPerTile:
				return "Gaussian Count Per Tile";
			case DebugVisualization::GaussianCountPerPixel:
				return "Gaussian Count Per Pixel";
			case DebugVisualization::Transmittance:
				return "Transmittance";
			case DebugVisualization::Depth:
				return "Depth";
			case DebugVisualization::Opacity:
				return "Opacity";
			case DebugVisualization::Distortion:
				return "Distortion";
			case DebugVisualization::EXTENT_LOSS:
				return "Opacity Field Loss";
			case DebugVisualization::Depth_Index:
				return "Depth Index";
			case DebugVisualization::DEPTH_EXTENT:
				return "Depth Extent";
			default:
				return "Disabled";
		}
	};

	enum SortMode
	{
		GLOBAL = 0,
		PER_PIXEL_FULL = 1,
		PER_PIXEL_KBUFFER = 2,
		HIERARCHICAL = 3
	};

	enum GlobalSortOrder
	{
		VIEWSPACE_Z = 0,
		DISTANCE = 1,
		PER_TILE_DEPTH_CENTER = 2,
		PER_TILE_DEPTH_MAXPOS = 3,
		MIN_VIEWSPACE_Z = 4
	};

	struct SortQueueSizes
	{
		int tile_4x4 = 64;
		int tile_2x2 = 8;
		int per_pixel = 4;
	};

#ifdef STOPTHEPOP_FASTBUILD
	static const std::vector<int> per_pixel_queue_sizes{ 16 };
	static const std::vector<int> twobytwo_tile_queue_sizes{8};
	static const std::vector<int> per_pixel_queue_sizes_hier{4};
#else // STOPTHEPOP_FASTBUILD
	static const std::vector<int> per_pixel_queue_sizes{1, 2, 4, 8, 12, 16, 20, 24};
	static const std::vector<int> twobytwo_tile_queue_sizes{8, 12, 20};
	static const std::vector<int> per_pixel_queue_sizes_hier{4, 8, 16};
#endif // STOPTHEPOP_FASTBUILD

	struct SortSettings
	{
		SortMode sort_mode = SortMode::GLOBAL;
		GlobalSortOrder sort_order = GlobalSortOrder::VIEWSPACE_Z;
		SortQueueSizes queue_sizes;

		bool requiresDepthAlongRay() const
		{
			return sort_mode != SortMode::GLOBAL || 
				sort_order == GlobalSortOrder::PER_TILE_DEPTH_CENTER ||
				sort_order == GlobalSortOrder::PER_TILE_DEPTH_MAXPOS;
		}

		bool hasModifiableWindowSize()
		{
			return sort_mode == SortMode::HIERARCHICAL || sort_mode == SortMode::PER_PIXEL_KBUFFER;
		}
	};

	struct CullingSettings
	{
		bool rect_bounding = false;
		bool tight_opacity_bounding = false;
		bool tile_based_culling = false;
		bool hierarchical_4x4_culling = false;
	};

	struct MeshingSettings
	{
		bool alpha_early_stop = false;
		bool return_color = false;
		float transmittance_threshold = 0.5f;
	};

	inline std::string toString(SortMode m) {
		switch(m)
		{
			case SortMode::GLOBAL:
				return "GLOBAL";
			case SortMode::PER_PIXEL_FULL:
				return "FULL SORT";
			case SortMode::PER_PIXEL_KBUFFER:
				return "KBUFFER";
			case SortMode::HIERARCHICAL:
				return "HIERARCHICAL";
		}
	};

	inline bool isInvalidSortMode(int m){
		if (m < SortMode::GLOBAL || m > SortMode::HIERARCHICAL)
			return true;
		else
			return false;
	};

	inline std::string toString(GlobalSortOrder m) {
		switch(m)
		{
			case GlobalSortOrder::VIEWSPACE_Z:
				return "VIEWSPACE_Z";
			case GlobalSortOrder::DISTANCE:
				return "DISTANCE";
			case GlobalSortOrder::PER_TILE_DEPTH_CENTER:
				return "PER_TILE_DEPTH_CENTER";
			case GlobalSortOrder::PER_TILE_DEPTH_MAXPOS:
				return "PER_TILE_DEPTH_MAXPOS";
		}
	};

	inline bool isInvalidSortOrder(int m){
		if (m < GlobalSortOrder::VIEWSPACE_Z || m > GlobalSortOrder::PER_TILE_DEPTH_MAXPOS)
			return true;
		else
			return false;
	};

	struct SplattingSettings
    {
        SortSettings sort_settings;
        CullingSettings culling_settings;
		MeshingSettings meshing_settings;
        bool load_balancing;
        bool proper_ewa_scaling;
		// new stuff
        bool exact_depth;
        bool detach_alpha;
		float far_plane;
		bool detach_alpha_extent;
		bool include_alpha;
		bool render_opacity;
    };

	struct DebugVisualizationData
	{
		DebugVisualization type{DebugVisualization::Disabled};
		int debugX{0};
		int debugY{0};
	
		float min{0.f};
		float max{1.f};
		bool debug_normalize = false;
	
		bool timing_enabled = false;
		bool printing_enabled = true;
		bool colormap = true;
		int precision = 3;
	};

    void inline to_json(nlohmann::json& j, const SplattingSettings& s)
    {
        j = nlohmann::json{
            {"sort_settings", {
                {"sort_mode", s.sort_settings.sort_mode},
                {"sort_order", s.sort_settings.sort_order},
                {"queue_sizes", {
                    {"tile_4x4", s.sort_settings.queue_sizes.tile_4x4},
                    {"tile_2x2", s.sort_settings.queue_sizes.tile_2x2},
                    {"per_pixel", s.sort_settings.queue_sizes.per_pixel}
                }}
            }},
            {"culling_settings", {
                {"rect_bounding", s.culling_settings.rect_bounding},
                {"tight_opacity_bounding", s.culling_settings.tight_opacity_bounding},
                {"tile_based_culling", s.culling_settings.tile_based_culling},
                {"hierarchical_4x4_culling", s.culling_settings.hierarchical_4x4_culling}
            }},
            {"meshing_settings", {
                {"alpha_early_stop", s.meshing_settings.alpha_early_stop},
                {"return_color", s.meshing_settings.return_color},
                {"transmittance_threshold", s.meshing_settings.transmittance_threshold},
            }},
            {"load_balancing", s.load_balancing},
            {"proper_ewa_scaling", s.proper_ewa_scaling},
            {"exact_depth", s.exact_depth},
            {"detach_alpha", s.detach_alpha},
            {"far_plane", s.far_plane},
            {"detach_alpha_extent", s.detach_alpha_extent},
            {"include_alpha", s.include_alpha},
            {"render_opacity", s.render_opacity},
        };
    }

    void inline from_json(const nlohmann::json& j, SplattingSettings& s)
    {
        {
            const nlohmann::json& j_sort = j["sort_settings"];
            j_sort.at("sort_mode").get_to(s.sort_settings.sort_mode);
            j_sort.at("sort_order").get_to(s.sort_settings.sort_order);
            {
                const nlohmann::json& j_sort_queue = j_sort["queue_sizes"];
                j_sort_queue.at("tile_4x4").get_to(s.sort_settings.queue_sizes.tile_4x4);
                j_sort_queue.at("tile_2x2").get_to(s.sort_settings.queue_sizes.tile_2x2);
                j_sort_queue.at("per_pixel").get_to(s.sort_settings.queue_sizes.per_pixel);
            }
        }
        {
            const nlohmann::json& j_culling = j["culling_settings"];
            j_culling.at("rect_bounding").get_to(s.culling_settings.rect_bounding);
            j_culling.at("tight_opacity_bounding").get_to(s.culling_settings.tight_opacity_bounding);
            j_culling.at("tile_based_culling").get_to(s.culling_settings.tile_based_culling);
            j_culling.at("hierarchical_4x4_culling").get_to(s.culling_settings.hierarchical_4x4_culling);
        }
        {
            const nlohmann::json& j_meshing = j["meshing_settings"];
            j_meshing.at("alpha_early_stop").get_to(s.meshing_settings.alpha_early_stop);
            j_meshing.at("return_color").get_to(s.meshing_settings.return_color);
			if (j_meshing.contains("transmittance_threshold"))
				j_meshing.at("transmittance_threshold").get_to(s.meshing_settings.transmittance_threshold);
        }
        j.at("load_balancing").get_to(s.load_balancing);
        j.at("proper_ewa_scaling").get_to(s.proper_ewa_scaling);
        j.at("exact_depth").get_to(s.exact_depth);
        j.at("detach_alpha").get_to(s.detach_alpha);
        j.at("far_plane").get_to(s.far_plane);
        j.at("detach_alpha_extent").get_to(s.detach_alpha_extent);
        j.at("include_alpha").get_to(s.include_alpha);
        j.at("render_opacity").get_to(s.render_opacity);
    }

    void inline to_json(nlohmann::json& j, const DebugVisualizationData& d)
    {
        j = nlohmann::json{
            {"type", d.type},
            {"debugX", d.debugX},
            {"debugY", d.debugY},
            {"min", d.min},
            {"max", d.max},
            {"debug_normalize", d.debug_normalize},
            {"timing_enabled", d.timing_enabled},
            {"printing_enabled", d.printing_enabled},
            {"colormap", d.colormap},
            {"precision", d.precision},
        };
    }

	void inline from_json(const nlohmann::json& j, DebugVisualizationData& d)
    {
        j.at("type").get_to(d.type);
		// debug pixel
        j.at("debugX").get_to(d.debugX);
        j.at("debugY").get_to(d.debugY);
		// manual normalization
        j.at("min").get_to(d.min);
        j.at("max").get_to(d.max);
		j.at("debug_normalize").get_to(d.debug_normalize);

		j.at("timing_enabled").get_to(d.timing_enabled);
		j.at("printing_enabled").get_to(d.printing_enabled);
		j.at("colormap").get_to(d.colormap);
		j.at("precision").get_to(d.precision);
    }

	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static void Compute3DFilter(
			int P, // num_points
			int C, // num_cameras
			const float* means3D,
			const float* viewmatrices,
			const int W, const int H,
			const float focal_x, const float focal_y,
			float* filter_3D
		);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const SplattingSettings splatting_settings,
			DebugVisualizationData& debugVisualization,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* view2gaussian_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* inv_viewprojmatrix,
			const float* filter_3d,
			const float* cam_pos,
			const float tan_fovx, const float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const SplattingSettings splatting_settings,
			const float* means3D,
			const float* shs,
			const float* opacities,
			const float* colors_precomp,
			const float* view2gaussian_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* inv_viewprojmatrix,
			const float* filter_3d,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const float* pixel_colors,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dview2gaussian,
			bool debug);
		
		static int integrate(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> pointBuffer,
			std::function<char* (size_t)> point_binningBuffer,
			const int PN, const int P, int D, int M,
			const float* background,
			const int width, int height,
			const SplattingSettings splatting_settings,
			DebugVisualizationData& debugVisualization,
			const float* points3D,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* view2gaussian_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* inv_viewprojmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii = nullptr,
			float* out_alpha_integrated = nullptr,
			float* out_color_integrated = nullptr,
			bool debug = false);

		static int computeTransmittance(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> pointBuffer,
			std::function<char* (size_t)> point_binningBuffer,
			const int PN, const int P, int D, int M,
			const int width, int height,
			const SplattingSettings splatting_settings,
			const float* points3D,
			const float* means3D,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* view2gaussian_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* inv_viewprojmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			int* radii,
			float* out_transmittance,
			bool debug = false);
	};
	

};



#endif